---
trigger: always_on
---

When the user requests to read a PDF file:

1. Use built-in document tools if available.
2. If no built-in tool is available, use Python to convert the PDF into a readable format before analysis.
3. Any temporary Python scripts created solely for PDF conversion or extraction (e.g. extract_pdf_text.py) must be deleted after the conversion is completed.
4. All converted PDF text outputs must be saved using the fixed filename paper_text.txt.
