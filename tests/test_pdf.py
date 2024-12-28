import pytest
from langchain.document_loaders import PyPDFLoader  # Assuming PyPDFLoader is the correct class import
import os

def test_pdf_loading_and_parsing():
    """Test loading and parsing a PDF file using PyPDFLoader."""
    # Ensure the test PDF file exists
    pdf_path = "./test_data/testpaper.pdf"
    assert os.path.exists(pdf_path), f"PDF file {pdf_path} does not exist."

    # Load the PDF using PyPDFLoader
    pdf_loader = PyPDFLoader(pdf_path)
    pdf_pages = pdf_loader.load()

    # Check if pdf_pages is a list
    assert isinstance(pdf_pages, list), f"Expected a list, but got {type(pdf_pages)}"

    # Ensure each page is a document object (checking type)
    for page in pdf_pages:
        assert hasattr(page, "metadata"), f"Page object is missing 'metadata' attribute."
        assert hasattr(page, "content"), f"Page object is missing 'content' attribute."

    # Check if the metadata contains expected information
    for page in pdf_pages:
        assert "source" in page.metadata, f"Missing 'source' in metadata for page."
        assert "page_number" in page.metadata, f"Missing 'page_number' in metadata for page."

    # Basic content splitting test (first page content)
    first_page_content = pdf_pages[0].content  # Assuming 'content' holds the text of the page
    assert isinstance(first_page_content, str), "Page content is not a string."
    
    # Split content into words or sentences for further basic validation
    content_words = first_page_content.split()
    assert len(content_words) > 0, "First page content appears empty after splitting."

    # Optionally, print first 100 words from first page content for debugging
    print("First 100 words of first page content:", " ".join(content_words[:100]))

    # You could also check specific text if known to be in the PDF:
    # assert "Expected phrase" in first_page_content, "Expected phrase not found in the first page content."
