# 🏥 PubMed RAG: Medical Literature Analysis with BigQuery and Gemini

This repository contains notebooks and resources that demonstrate how to build RAG (Retrieval-Augmented Generation) applications for medical literature analysis using Google Cloud BigQuery vector search and Vertex AI Gemini models.

The project converts the user experience from the [Capricorn Medical Research Application](https://capricorn-medical-research.web.app/) into interactive Colab notebooks, making it accessible for both clinicians and data scientists.

## Quick Start Guide

### 🚀 For Clinicians (No Coding Required)

1. Open the [Clinician Example notebook](PubMed_RAG_Clinician_Example.ipynb)
2. Click **Runtime → Run all** (or press Ctrl/Cmd + F9)
3. Authenticate with your Google account
4. Use the interactive Gradio app to:
   - Paste your medical case notes
   - Extract disease and events automatically
   - Search and analyze PubMed literature
   - Generate comprehensive analysis reports

### 💻 For Data Scientists

1. Open the [Data Scientist Example notebook](PubMed_RAG_Data_Scientist_Example.ipynb)
2. Configure your Google Cloud project
3. Customize the analysis pipeline:
   ```python
   # Define custom scoring criteria
   CUSTOM_CRITERIA = [
       {"name": "clinical_trial", "weight": 50},
       {"name": "pediatric_focus", "weight": 60},
       # Add your own criteria
   ]
   
   # Process medical case
   results = process_medical_case(
       case_text,
       default_articles=10,
       min_per_event=3
   )
   ```

## Architecture

![Medical Literature Analysis Architecture](https://github.com/google/pubmed-rag/blob/main/visuals/1.png?raw=true)

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
