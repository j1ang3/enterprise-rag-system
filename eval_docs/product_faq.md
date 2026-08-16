# Acme Knowledge Assistant FAQ

## Supported Sources

Acme Knowledge Assistant can ingest Markdown, plain text, PDF files with embedded text, and DOCX documents. Scanned PDFs are not supported unless OCR is performed before upload.

## Synchronization

The system checks connected source folders every 4 hours. Manual synchronization can be started by an administrator from the admin console. Deleted source files are removed from search results after the next successful sync.

## Answer Citations

Every generated answer should include citations to the source chunks used. If no relevant source is found, the assistant should say that the knowledge base does not contain a reliable answer.

## Access Control

Search results must respect document permissions. Users should only retrieve chunks from documents they are allowed to view. Permission changes are applied during the next synchronization cycle.

## Limits

The recommended single document size limit is 25 MB. The recommended maximum chunk size is 800 tokens. Very large documents should be split by section before upload when possible.
