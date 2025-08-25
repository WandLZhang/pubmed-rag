# 🏥 PubMed RAG: Medical Literature Analysis with BigQuery and Gemini

This repository contains notebooks and resources that demonstrate how to build RAG (Retrieval-Augmented Generation) applications for medical literature analysis using Google Cloud BigQuery vector search and Vertex AI Gemini models.

## Overview

[PubMed RAG](https://github.com/WandLZhang/pubmed-rag) provides a complete pipeline for analyzing medical cases against PubMed literature using advanced AI capabilities. This repository demonstrates how to extract medical information from case notes, search relevant PubMed articles using vector similarity, score and rank articles with customizable criteria, and generate evidence-based analyses with proper citations.

The project converts the user experience from the [Capricorn Medical Research Application](https://capricorn-medical-research.web.app/) into interactive Colab notebooks, making it accessible for both clinicians and data scientists.

## Features

- 🔬 **Medical Information Extraction**: Automatically extract disease diagnoses and actionable events from clinical case notes
- 🔍 **Vector Search**: Search PubMed literature using BigQuery vector similarity with text embeddings
- 📊 **Dynamic Scoring**: Customizable scoring criteria for ranking articles based on relevance, impact, and quality
- 🤖 **AI-Powered Analysis**: Generate comprehensive literature syntheses using Gemini models
- 💬 **Interactive Chat**: Ask questions about medical cases and get evidence-based answers
- 📈 **Two-Phase Analysis**: Progressive search with event coverage tracking for comprehensive results

## Explore, learn and contribute

You can explore, learn, and contribute to this repository to advance medical literature analysis with AI!

### Explore and learn

Explore this repository and follow the links in the header section of each notebook to:

![Colab](https://cloud.google.com/ml-engine/images/colab-logo-32px.png) Open and run the notebook in [Colab](https://colab.google/)  
![Colab Enterprise](https://cloud.google.com/ml-engine/images/colab-enterprise-logo-32px.png) Open and run the notebook in [Colab Enterprise](https://cloud.google.com/colab/docs/introduction)  
![Workbench](https://lh3.googleusercontent.com/UiNooY4LUgW_oTvpsNhPpQzsstV5W8F7rYgxgGBD85cWJoLmrOzhVs_ksK_vgx40SHs7jCqkTkCk=e14-rj-sc0xffffff-h130-w32) Open and run the notebook in [Vertex AI Workbench](https://cloud.google.com/vertex-ai/docs/workbench/introduction)  
![Github](https://cloud.google.com/ml-engine/images/github-logo-32px.png) View the notebook on GitHub

### Contribute

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Get started

To get started with PubMed RAG, you need:

1. **A Google Cloud Project** with billing enabled
   - If you don't have one, you can use the [Free Trial](https://cloud.google.com/free) with $300 credit
   - Learn more about [setting up a project](https://cloud.google.com/vertex-ai/docs/start/cloud-environment)

2. **Enable Required APIs**:
   - Vertex AI API
   - BigQuery API
   - Cloud Resource Manager API

3. **Choose your environment**:
   - **Quick Start**: Click "Run in Colab" on any notebook for immediate access
   - **Enterprise**: Use Colab Enterprise for enhanced security and collaboration
   - **Local Development**: Clone the repository and run in Vertex AI Workbench

## Examples

<!-- markdownlint-disable MD033 -->
<table>
  <tr>
    <th style="text-align: center;">Notebook</th>
    <th style="text-align: center;">Target Audience</th>
    <th style="text-align: center;">Description</th>
    <th style="text-align: center;">Key Features</th>
  </tr>
  <tr>
    <td>
      <a href="PubMed_RAG_Clinician_Example.ipynb"><code>Clinician Example</code></a>
    </td>
    <td>Healthcare Professionals</td>
    <td>
      Interactive Gradio app for medical literature analysis without coding
    </td>
    <td>
      • No-code interface<br>
      • Case note analysis<br>
      • Customizable scoring<br>
      • Final synthesis generation<br>
      • Two-phase progressive search
    </td>
  </tr>
  <tr>
    <td>
      <a href="PubMed_RAG_Data_Scientist_Example.ipynb"><code>Data Scientist Example</code></a>
    </td>
    <td>Researchers & Developers</td>
    <td>
      Complete code pipeline for building custom medical RAG applications
    </td>
    <td>
      • Full Python implementation<br>
      • Customizable prompts<br>
      • Dynamic scoring system<br>
      • Interactive visualizations<br>
      • Medical Q&A chat
    </td>
  </tr>
</table>
<!-- markdownlint-enable MD033 -->

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

![Medical Literature Analysis Architecture](https://github.com/WandLZhang/pubmed-rag/blob/main/visuals/1.png?raw=true)

The system uses:
- **BigQuery**: Vector database for PubMed article embeddings
- **Vertex AI Gemini**: LLM for information extraction and analysis
- **Text Embeddings**: For semantic similarity search
- **Dynamic Scoring**: Customizable relevance ranking

## Key Components

### 1. Medical Information Extraction
Extract structured information from unstructured case notes:
- Primary disease diagnosis
- Actionable events (treatments, mutations, complications)
- Clinical concepts for literature search

### 2. Vector Search Pipeline
- Generates embeddings for case queries
- Searches PubMed database using BigQuery vector similarity
- Returns most relevant articles based on semantic matching

### 3. Article Scoring System
Dynamic scoring based on:
- Journal impact factor (SJR)
- Publication recency
- Event matching
- Study type and quality
- Custom criteria

### 4. Literature Synthesis
Generate comprehensive analyses including:
- Executive summaries
- Key findings by concept
- Methodological landscape
- Knowledge gaps
- Evidence-based recommendations

## Technologies Used

- **Google Cloud Platform**
  - Vertex AI (Gemini models)
  - BigQuery (Vector search)
  - Cloud Storage
- **Python Libraries**
  - `google-genai`: Gemini SDK
  - `google-cloud-bigquery`: BigQuery client
  - `gradio`: Interactive web interfaces
  - `plotly`: Data visualization
  - `pandas`: Data manipulation

## Important Notice

⚠️ **For Research Purposes Only**

This tool is a **DEMONSTRATION** showcasing AI-powered research capabilities:
- For research and educational purposes only
- NOT intended for treatment planning or clinical decisions
- All AI-generated analyses should be verified against primary sources
- Results may contain inaccuracies or limitations

## Authors

- [Willis Zhang](https://github.com/WandLZhang)
- [Stone Jiang](https://github.com/siduojiang)

## Get Help

- **Issues**: Use the [Issues page](https://github.com/WandLZhang/pubmed-rag/issues) to report bugs or request features
- **Discussions**: Join conversations about medical RAG applications
- **Documentation**: Check the notebooks for detailed implementation guides

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This is not an officially supported Google product. The code in this repository is for demonstrative purposes only. This tool should not be used for making clinical decisions or treatment planning.

## References

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [BigQuery Vector Search](https://cloud.google.com/bigquery/docs/vector-search)
- [Gemini API Documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/gemini-api-overview)
- [PubMed Central](https://www.ncbi.nlm.nih.gov/pmc/)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{pubmed_rag_2025,
  author = {Zhang, Willis and Jiang, Stone},
  title = {PubMed RAG: Medical Literature Analysis with BigQuery and Gemini},
  year = {2025},
  url = {https://github.com/WandLZhang/pubmed-rag}
}
