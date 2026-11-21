"""
Resume text extraction and PII masking.

extract_text() reads a resume from a .pdf, .docx, .txt or .md file (a path, or bytes plus a file name) and returns clean text.
redact_pii() masks emails, links, phone numbers and ID numbers (Aadhaar, PAN, US SSN formats), and any names you pass in.

What it does not do: it does not mask names or street addresses unless you pass the names, it does not read scanned PDFs (no OCR),
and it does not anonymise anything, it only reduces what leaves the machine. DOCX files are read with the standard library
(no extra package); PDF files need pypdf.

Usage:
    python resume_parser.py resume.pdf
    python resume_parser.py resume.docx --redact --name "Asha Verma"
"""
import argparse
import io
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
MAX_PAGES = 15
MAX_UNZIPPED = 50 * 1024 * 1024
SUPPORTED = (".pdf", ".docx", ".txt", ".md")
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


class ResumeParseError(ValueError):
    """The file could not be turned into resume text; the message is safe to show to a user."""


_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)


def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ").translate(_ZERO_WIDTH)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _size(limit):
    return "%d MB" % (limit // (1024 * 1024)) if limit >= 1024 * 1024 else "%d bytes" % limit


def _decode(data):
    if b"\x00" in data[:4096]:
        raise ResumeParseError("This looks like a binary file, not text. Save the resume as UTF-8 text, PDF or DOCX.")
    for enc in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _pdf_text(data):
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise ResumeParseError("PDF support needs the pypdf package (pip install pypdf).") from e
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise ResumeParseError("The PDF is password protected.")
        n = len(reader.pages)
        if n > MAX_PAGES:
            raise ResumeParseError("The PDF has %d pages; the limit is %d." % (n, MAX_PAGES))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except ResumeParseError:
        raise
    except Exception as e:
        raise ResumeParseError("Could not read the PDF (%s)." % type(e).__name__) from e
    if not clean_text(text):
        raise ResumeParseError("The PDF has no extractable text (it may be a scan; OCR is not supported).")
    return text


def _para_text(p):
    parts = []
    for el in p.iter():
        if el.tag == W + "t":
            parts.append(el.text or "")
        elif el.tag == W + "tab":
            parts.append("\t")
        elif el.tag in (W + "br", W + "cr"):
            parts.append("\n")
    return "".join(parts)


def _docx_text(data):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ResumeParseError("The file is not a valid DOCX (it is not a zip archive).") from e
    with zf:
        if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED:
            raise ResumeParseError("The DOCX expands to more than %s." % _size(MAX_UNZIPPED))
        if "word/document.xml" not in zf.namelist():
            raise ResumeParseError("The file is not a Word document (word/document.xml is missing).")
        xml = zf.read("word/document.xml")
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise ResumeParseError("The DOCX contains unsupported XML declarations.")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as e:
        raise ResumeParseError("The DOCX could not be read (invalid XML).") from e
    body = root.find(W + "body")
    if body is None:
        raise ResumeParseError("The DOCX has no body.")
    lines = []

    def walk(container):
        for child in container:
            if child.tag == W + "p":
                lines.append(_para_text(child))
            elif child.tag == W + "tbl":
                for tr in child.iter(W + "tr"):
                    cells = []
                    for tc in tr.findall(W + "tc"):
                        t = " ".join(_para_text(p).strip() for p in tc.iter(W + "p")).strip()
                        if t and (not cells or cells[-1] != t):
                            cells.append(t)
                    if cells:
                        lines.append(" | ".join(cells))
            elif child.tag == W + "sdt":
                content = child.find(W + "sdtContent")
                if content is not None:
                    walk(content)

    walk(body)
    return "\n".join(lines)


def extract_text(source, filename=None, max_bytes=None):
    """Text of a resume file. source is a path, or bytes (then filename tells the type; without one the type is sniffed)."""
    limit = MAX_BYTES if max_bytes is None else max_bytes
    if isinstance(source, (bytes, bytearray)):
        data, name = bytes(source), filename or ""
    else:
        path = Path(source)
        name = filename or path.name
        try:
            size = path.stat().st_size
        except OSError as e:
            raise ResumeParseError("Could not open the file: %s" % e) from e
        if size > limit:
            raise ResumeParseError("The file is larger than the allowed size (%s)." % _size(limit))
        data = path.read_bytes()
    if len(data) > limit:
        raise ResumeParseError("The file is larger than the allowed size (%s)." % _size(limit))
    if not data:
        raise ResumeParseError("The file is empty.")
    ext = Path(name).suffix.lower()
    if not ext:
        ext = ".pdf" if data.startswith(b"%PDF") else (".docx" if data.startswith(b"PK") else ".txt")
    if ext not in SUPPORTED:
        raise ResumeParseError("Unsupported file type %r. Use PDF, DOCX or a text file." % ext)
    if ext == ".pdf":
        if b"%PDF" not in data[:1024]:
            raise ResumeParseError("The file is not a PDF.")
        text = _pdf_text(data)
    elif ext == ".docx":
        text = _docx_text(data)
    else:
        text = _decode(data)
    text = clean_text(text)
    if not text:
        raise ResumeParseError("No text was found in the file.")
    return text


_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\"']+")
_SITE = re.compile(r"(?i)\b(?:linkedin\.com|github\.com|gitlab\.com|bitbucket\.org|kaggle\.com|leetcode\.com|medium\.com)/[^\s<>\"']*")
_AADHAAR = re.compile(r"(?<!\d)\d{4}[ \-]\d{4}[ \-]\d{4}(?!\d)")
_SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
_PAN = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")
_PHONE = re.compile(r"(?<![\w.])\+?\(?\d[\d \t().\-]{8,}\d(?![\w])")


def redact_pii(text, names=None):
    """(masked text, counts by kind). Kinds: name, email, link, id_number, phone. Applying it twice changes nothing more."""
    counts = {}

    def bump(kind, n=1):
        counts[kind] = counts.get(kind, 0) + n

    out = text
    for name in sorted({n.strip() for n in (names or []) if n and n.strip()}, key=len, reverse=True):
        out, n = re.subn(re.escape(name), "[NAME]", out, flags=re.IGNORECASE)
        if n:
            bump("name", n)
    out, n = _EMAIL.subn("[EMAIL]", out)
    if n:
        bump("email", n)

    def link_sub(m):
        s = m.group(0)
        core = s.rstrip(".,;:!?)]}\"'")
        bump("link")
        return "[LINK]" + s[len(core):]

    out = _URL.sub(link_sub, out)
    out = _SITE.sub(link_sub, out)
    def id_sub(m):
        groups = re.findall(r"\d+", m.group(0))
        if len(groups) == 3 and all(len(g) == 4 and g[:2] in ("19", "20") for g in groups):
            return m.group(0)  # a run of years such as 2018-2020 2021, not an Aadhaar number
        bump("id_number")
        return "[ID NUMBER]"

    for pat in (_AADHAAR, _SSN, _PAN):
        out = pat.sub(id_sub, out)

    def phone_sub(m):
        s = m.group(0)
        groups = re.findall(r"\d+", s)
        if not 10 <= sum(len(g) for g in groups) <= 15:
            return s
        if max(len(g) for g in groups) < 3 or len(groups) > 6:
            return s  # a row of short numbers, not a phone number
        if all(len(g) == 4 and g[:2] in ("19", "20") for g in groups):
            return s  # a run of years
        bump("phone")
        return "[PHONE]"

    out = _PHONE.sub(phone_sub, out)
    return out, counts


def main(argv=None):
    ap = argparse.ArgumentParser(description="Extract (and optionally mask) the text of a resume file")
    ap.add_argument("file")
    ap.add_argument("--redact", action="store_true", help="mask emails, links, phone numbers and ID numbers")
    ap.add_argument("--name", action="append", default=[], help="a name to mask (repeat for several)")
    args = ap.parse_args(argv)
    try:
        text = extract_text(args.file)
    except ResumeParseError as e:
        raise SystemExit("Error: %s" % e)
    if args.redact or args.name:
        text, counts = redact_pii(text, args.name)
        print("Masked: " + (", ".join("%s %d" % (k, v) for k, v in counts.items()) or "nothing found"), file=sys.stderr)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
