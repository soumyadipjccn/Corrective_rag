"""
Unit tests for data schemas and models.
"""

from src.schemas.state import DocumentGrade, IngestionResult, FormattedDocument


def test_document_grade_valid():
    grade_yes = DocumentGrade(score="yes")
    assert grade_yes.score == "yes"

    grade_no = DocumentGrade(score="no", explanation="Off-topic content")
    assert grade_no.score == "no"
    assert grade_no.explanation == "Off-topic content"


def test_ingestion_result():
    res = IngestionResult(
        source="test.pdf",
        doc_count=5,
        chunk_count=15,
        collection_name="test-coll",
        vector_dim=1024,
        status="success"
    )
    assert res.doc_count == 5
    assert res.chunk_count == 15
    assert res.vector_dim == 1024
    assert res.status == "success"


def test_formatted_document():
    doc_info = FormattedDocument(
        source="arxiv.pdf",
        title="Llama 2 Paper",
        snippet="Pretrained LLMs"
    )
    assert doc_info.source == "arxiv.pdf"
    assert doc_info.title == "Llama 2 Paper"
