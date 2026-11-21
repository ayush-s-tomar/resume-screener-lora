"""Tests for resume_parser: file extraction (text, DOCX, PDF) and PII masking."""
import io
import sys
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import resume_parser as rp  # noqa: E402

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>'
)
RELS = (
    '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'
)


def make_docx(paragraphs=(), table_rows=None, body_xml=None, prefix=""):
    def p(text):
        return '<w:p><w:r><w:t xml:space="preserve">%s</w:t></w:r></w:p>' % escape(text)

    body = body_xml if body_xml is not None else "".join(p(t) for t in paragraphs)
    if table_rows:
        rows = "".join("<w:tr>" + "".join("<w:tc>%s</w:tc>" % p(c) for c in row) + "</w:tr>" for row in table_rows)
        body += "<w:tbl>%s</w:tbl>" % rows
    xml = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>%s<w:document xmlns:w="%s"><w:body>%s</w:body></w:document>' % (prefix, W_NS, body)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CONTENT_TYPES)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", xml)
    return buf.getvalue()


def make_pdf(text):
    esc = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = "BT /F1 12 Tf 72 720 Td (%s) Tj ET" % esc
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        "<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += ("%d 0 obj\n%s\nendobj\n" % (i, body)).encode("latin-1")
    xref = len(out)
    out += ("xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)).encode("latin-1")
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode("latin-1")
    out += ("trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)).encode("latin-1")
    return out


# ---------- clean_text and plain text ----------

def test_clean_text():
    raw = "Line one  \r\n\r\n\r\n\r\nLine\u00a0two\u200b\x00\n"
    assert rp.clean_text(raw) == "Line one\n\nLine two"


def test_text_decoding():
    assert rp.extract_text("é Python".encode("utf-8-sig"), "r.txt") == "é Python"
    assert rp.extract_text("café".encode("cp1252"), "r.txt") == "café"


def test_path_input(tmp_path):
    f = tmp_path / "resume.md"
    f.write_text("Backend developer\n\nPython", encoding="utf-8")
    assert rp.extract_text(f) == "Backend developer\n\nPython"
    assert rp.extract_text(str(f)) == "Backend developer\n\nPython"


def test_type_is_sniffed_without_an_extension():
    assert rp.extract_text(b"plain text") == "plain text"
    assert rp.extract_text(make_docx(["Hello docx"])) == "Hello docx"


@pytest.mark.parametrize("data,name,match", [
    (b"", "r.txt", "empty"),
    (b"   \n  ", "r.txt", "No text"),
    (b"abc\x00def", "r.txt", "binary"),
    (b"x", "r.exe", "Unsupported"),
    (b"hello", "r.pdf", "not a PDF"),
    (b"hello", "r.docx", "not a valid DOCX"),
])
def test_errors(data, name, match):
    with pytest.raises(rp.ResumeParseError, match=match):
        rp.extract_text(data, name)


def test_size_limit():
    with pytest.raises(rp.ResumeParseError, match="larger than"):
        rp.extract_text(b"a" * 11, "r.txt", max_bytes=10)


def test_size_limit_on_a_path(tmp_path):
    f = tmp_path / "big.txt"
    f.write_bytes(b"a" * 11)
    with pytest.raises(rp.ResumeParseError, match="larger than"):
        rp.extract_text(f, max_bytes=10)


def test_missing_file(tmp_path):
    with pytest.raises(rp.ResumeParseError, match="Could not open"):
        rp.extract_text(tmp_path / "nope.txt")


# ---------- DOCX ----------

def test_docx_paragraphs_and_table_in_order():
    data = make_docx(["Asha Verma", "Backend developer"], table_rows=[["Skills", "Python, FastAPI"], ["Years", "4"]])
    assert rp.extract_text(data, "r.docx") == "Asha Verma\nBackend developer\nSkills | Python, FastAPI\nYears | 4"


def test_docx_merged_cells_are_not_repeated():
    data = make_docx([], table_rows=[["Summary", "Summary", "Backend developer"]])
    assert rp.extract_text(data, "r.docx") == "Summary | Backend developer"


def test_docx_tabs_and_breaks():
    body = '<w:p><w:r><w:t>A</w:t><w:tab/><w:t>B</w:t><w:br/><w:t>C</w:t></w:r></w:p>'
    assert rp.extract_text(make_docx(body_xml=body), "r.docx") == "A\tB\nC"


def test_docx_deleted_text_is_ignored():
    body = '<w:p><w:r><w:t>Kept</w:t></w:r><w:del><w:r><w:delText>Removed</w:delText></w:r></w:del></w:p>'
    assert rp.extract_text(make_docx(body_xml=body), "r.docx") == "Kept"


def test_docx_without_a_document_part():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("hello.txt", "hi")
    with pytest.raises(rp.ResumeParseError, match="not a Word document"):
        rp.extract_text(buf.getvalue(), "r.docx")


def test_docx_unzipped_size_limit(monkeypatch):
    monkeypatch.setattr(rp, "MAX_UNZIPPED", 100)
    with pytest.raises(rp.ResumeParseError, match="expands to more than"):
        rp.extract_text(make_docx(["Hello"]), "r.docx")


def test_docx_entity_declarations_are_rejected():
    data = make_docx(["Hello"], prefix='<!DOCTYPE d [<!ENTITY x "y">]>')
    with pytest.raises(rp.ResumeParseError, match="unsupported XML"):
        rp.extract_text(data, "r.docx")


# ---------- PDF (needs pypdf; skipped when it is not installed) ----------

def test_pdf_text():
    pytest.importorskip("pypdf")
    out = rp.extract_text(make_pdf("Python developer with 4 years of experience (FastAPI)"), "r.pdf")
    assert "Python developer with 4 years of experience (FastAPI)" in out


def test_pdf_without_text():
    pytest.importorskip("pypdf")
    with pytest.raises(rp.ResumeParseError, match="no extractable text"):
        rp.extract_text(make_pdf(""), "r.pdf")


def test_pdf_page_limit(monkeypatch):
    pytest.importorskip("pypdf")
    monkeypatch.setattr(rp, "MAX_PAGES", 0)
    with pytest.raises(rp.ResumeParseError, match="limit is 0"):
        rp.extract_text(make_pdf("Python"), "r.pdf")


def test_broken_pdf_gives_a_clean_error():
    pytest.importorskip("pypdf")
    with pytest.raises(rp.ResumeParseError):
        rp.extract_text(b"%PDF-1.4 this is not a real pdf", "r.pdf")


# ---------- masking ----------

def test_email_and_links():
    text = "Contact: john.doe+jobs@example.co.in, JANE@X.ORG. Profile: https://linkedin.com/in/john-doe. Code: github.com/johndoe and www.example.com/me)"
    out, counts = rp.redact_pii(text)
    assert "@" not in out and "linkedin" not in out.lower() and "github" not in out.lower() and "example.com" not in out
    assert counts == {"email": 2, "link": 3}
    assert "[LINK]." in out and "[LINK])" in out


@pytest.mark.parametrize("number", ["+91 98765 43210", "+91-98765-43210", "(555) 123-4567", "555-123-4567", "98765-43210", "+44 20 7946 0958"])
def test_phone_numbers(number):
    out, counts = rp.redact_pii("Call " + number + " today")
    assert out == "Call [PHONE] today"
    assert counts == {"phone": 1}


def test_years_skills_and_numbers_are_kept():
    text = "Python 3.11, Node.js, ASP.NET, C++, 2018 - 2022, 2019-2023, 2018-2020 2021, 10 LPA, 4.5 years, scores 90 85 78 92 88, GPA 8.7/10"
    out, counts = rp.redact_pii(text)
    assert out == text
    assert counts == {}


def test_id_numbers():
    out, counts = rp.redact_pii("Aadhaar 1234 5678 9012, PAN ABCDE1234F, SSN 123-45-6789")
    assert out == "Aadhaar [ID NUMBER], PAN [ID NUMBER], SSN [ID NUMBER]"
    assert counts == {"id_number": 3}


def test_names_are_masked_only_when_given():
    text = "Ayush Singh Tomar is a backend developer. ayush singh tomar"
    assert rp.redact_pii(text)[0] == text
    out, counts = rp.redact_pii(text, names=["Ayush Singh Tomar"])
    assert out == "[NAME] is a backend developer. [NAME]"
    assert counts == {"name": 2}


def test_masking_twice_changes_nothing_more():
    once, _ = rp.redact_pii("a@b.com +91 98765 43210 https://x.io/y 1234 5678 9012")
    twice, counts = rp.redact_pii(once)
    assert twice == once
    assert counts == {}


def test_multiline_numbers_are_not_joined():
    text = "Graduated 2020\n\n1234567\n2021"
    assert rp.redact_pii(text)[0] == text


def test_aadhaar_numbers_that_look_partly_like_years_are_still_masked():
    out, counts = rp.redact_pii("ID 2012 2034 5678")
    assert out == "ID [ID NUMBER]"
    assert counts == {"id_number": 1}
