"""
File format handlers for extracting text from multiple file types.

Supports: .txt, .docx, .xlsx, .pdf, .json
Each handler converts the file to extractable text that can be parsed
for SME gap responses containing [REF:gap-id]: answer patterns.
"""

from __future__ import annotations

import io
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False

try:
    from openpyxl import load_workbook
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False


class FileHandlerError(Exception):
    """Raised when file parsing fails."""
    pass


class BaseFileHandler(ABC):
    """Abstract base for file format handlers."""
    
    @abstractmethod
    def can_handle(self, filename: str) -> bool:
        """Return True if this handler can process the file."""
        pass
    
    @abstractmethod
    def extract_text(self, file_content: bytes) -> str:
        """Extract text from file content (bytes) and return as string."""
        pass


class PlainTextHandler(BaseFileHandler):
    """Handler for .txt files."""
    
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith('.txt')
    
    def extract_text(self, file_content: bytes) -> str:
        try:
            return file_content.decode('utf-8')
        except UnicodeDecodeError:
            return file_content.decode('latin-1', errors='ignore')


class DocxHandler(BaseFileHandler):
    """Handler for .docx (Word) files."""
    
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith('.docx')
    
    def extract_text(self, file_content: bytes) -> str:
        if not HAS_DOCX:
            raise FileHandlerError(
                "python-docx not installed. Install with: pip install python-docx"
            )
        
        try:
            doc = DocxDocument(io.BytesIO(file_content))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            
            # Also extract from tables
            table_texts = []
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(
                        cell.text.strip() for cell in row.cells
                    )
                    if row_text.strip():
                        table_texts.append(row_text)
            
            all_text = paragraphs + table_texts
            return '\n'.join(all_text)
        except Exception as e:
            raise FileHandlerError(f"Failed to parse .docx: {e}")


class ExcelHandler(BaseFileHandler):
    """Handler for .xlsx (Excel) files."""
    
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith(('.xlsx', '.xls'))
    
    def extract_text(self, file_content: bytes) -> str:
        if not HAS_OPENPYXL:
            raise FileHandlerError(
                "openpyxl not installed. Install with: pip install openpyxl"
            )
        
        try:
            workbook = load_workbook(io.BytesIO(file_content))
            lines = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                lines.append(f"=== Sheet: {sheet_name} ===")
                
                for row in sheet.iter_rows(values_only=True):
                    row_text = ' | '.join(
                        str(cell).strip() for cell in row if cell is not None
                    )
                    if row_text.strip():
                        lines.append(row_text)
            
            return '\n'.join(lines)
        except Exception as e:
            raise FileHandlerError(f"Failed to parse .xlsx: {e}")


class PdfHandler(BaseFileHandler):
    """Handler for .pdf files."""
    
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith('.pdf')
    
    def extract_text(self, file_content: bytes) -> str:
        if not HAS_PYPDF2:
            raise FileHandlerError(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )
        
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return '\n'.join(text)
        except Exception as e:
            raise FileHandlerError(f"Failed to parse .pdf: {e}")


class JsonHandler(BaseFileHandler):
    """Handler for .json files (structured responses)."""
    
    def can_handle(self, filename: str) -> bool:
        return filename.lower().endswith('.json')
    
    def extract_text(self, file_content: bytes) -> str:
        """Extract JSON and convert to readable text format."""
        try:
            data = json.loads(file_content.decode('utf-8'))
            return json.dumps(data, indent=2)
        except json.JSONDecodeError as e:
            raise FileHandlerError(f"Invalid JSON: {e}")
        except Exception as e:
            raise FileHandlerError(f"Failed to parse .json: {e}")


# Registry of handlers
_HANDLERS = [
    PlainTextHandler(),
    DocxHandler(),
    ExcelHandler(),
    PdfHandler(),
    JsonHandler(),
]


def get_handler(filename: str) -> Optional[BaseFileHandler]:
    """
    Get the appropriate handler for a filename.
    Returns None if no handler supports the file type.
    """
    for handler in _HANDLERS:
        if handler.can_handle(filename):
            return handler
    return None


def extract_text_from_file(filename: str, file_content: bytes) -> str:
    """
    Extract text from a file given its filename and binary content.
    
    Raises FileHandlerError if:
    - No handler exists for the file type
    - The file parsing fails
    """
    handler = get_handler(filename)
    if not handler:
        raise FileHandlerError(
            f"Unsupported file type: {Path(filename).suffix}. "
            f"Supported types: .txt, .docx, .xlsx, .pdf, .json"
        )
    
    return handler.extract_text(file_content)


def get_supported_formats() -> list[str]:
    """Return list of supported file extensions."""
    formats = set()
    formats.add('.txt')
    if HAS_DOCX:
        formats.add('.docx')
    if HAS_OPENPYXL:
        formats.add('.xlsx')
        formats.add('.xls')
    if HAS_PYPDF2:
        formats.add('.pdf')
    formats.add('.json')
    return sorted(list(formats))
