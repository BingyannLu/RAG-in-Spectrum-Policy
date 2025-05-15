# RAG-in-Spectrum-Policy
This repository provides an open-source spectrum policy dataset specifically designed for the development and evaluation of RAG systems. The dataset comprises 308 documents from four proceedings of the Federal Communications Commission (FCC) and the National Telecommunications and Information Administration (NTIA), totaling 3,098 pages.

The repository includes:

1. Knowledge Dataset: A comprehensive dataset to serve as the knowledge base for RAG systems. 
2. Test Dataset: A corresponding dataset of question-answer pairs for evaluating system performance.
Both datasets are provided in CSV format for ease of use and integration into various RAG workflows.
3. download_pdf.py is designed to automatically collect and archive public documents from FCC proceedings.
4. Spectrumrag.py is designed to run on Google Colab and integrates with Google Drive for data storage and retrieval. To run the program locally, some adjustments are needed to replace Colab-specific components and reconfigure file access paths.
