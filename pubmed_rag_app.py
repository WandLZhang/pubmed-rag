#!/usr/bin/env python3
"""
PubMed Medical Literature Analysis App
A Gradio application for analyzing medical cases using PubMed literature with BigQuery vector search and Gemini.
"""

import gradio as gr
import pandas as pd
import json
import math
from datetime import datetime
from google import genai
from google.genai import types
from google.cloud import bigquery
from google.cloud import resourcemanager_v3
from google.cloud import service_usage_v1
from google.cloud import billing_v1
from google.genai.types import GenerateContentConfig
import time
import os
import webbrowser
import argparse

# --- Constants ---
PUBMED_DATASET = "wz-data-catalog-demo.pubmed"
PUBMED_TABLE = f"{PUBMED_DATASET}.pmid_embed_nonzero_metadata"
MODEL_ID = "gemini-2.5-flash"  # Default model, will be updated dynamically
THINKING_BUDGET = 0  # Default thinking budget, will be updated dynamically
JOURNAL_IMPACT_CSV_URL = "https://raw.githubusercontent.com/WandLZhang/scimagojr_2024/main/scimagojr_2024.csv"
REQUIRED_APIS = ["aiplatform.googleapis.com", "bigquery.googleapis.com", "cloudresourcemanager.googleapis.com"]
CREATE_BILLING_ACCOUNT_URL = "https://console.cloud.google.com/billing/create?inv=1&invt=Ab4E_Q"
CREATE_BILLING_ACCOUNT_OPTION = "→ Create New Billing Account"
MODEL_OPTIONS = {
    "Gemini 2.5 Flash (Default)": "gemini-2.5-flash", 
    "Gemini 2.5 Pro": "gemini-2.5-pro"
}
SAMPLE_CASE = """A now almost 4-year-old female diagnosed with KMT2A-rearranged AML and CNS2 involvement exhibited refractory disease after NOPHO DBH AML 2012 protocol. Post- MEC and ADE, MRD remained at 35% and 53%. Vyxeos-clofarabine therapy reduced MRD to 18%. Third-line FLAG-Mylotarg lowered MRD to 3.5% (flow) and 1% (molecular). After a cord blood HSCT in December 2022, she relapsed 10 months later with 3% MRD and femoral extramedullary disease.
After the iLTB discussion, in November 2023 the patient was enrolled in the SNDX5613 trial, receiving revumenib for three months, leading to a reduction in KMT2A MRD to 0.1% by PCR. Subsequently, the patient underwent a second allogeneic HSCT using cord blood with treosulfan, thiotepa, and fludarabine conditioning, followed by revumenib maintenance. In August 2024, 6.5 months after the second HSCT, the patient experienced a bone marrow relapse with 33% blasts. The patient is currently in very good clinical condition.

Diagnostic tests:			
WES and RNAseq were performed on the 1st relapse sample showing KMT2A::MLLT3 fusion and NRAS (p.Gln61Lys) mutation.
Flow cytometry from the current relapse showed positive CD33 and CD123.
WES and RNAseq of the current relapse sample is pending."""

# --- Global Variables ---
genai_client, bq_client = None, None
journal_impact_dict = {}
PROJECT_ID = ""
LOCATION = "global"
USER_DATASET = "pubmed"

# Disease extraction prompt from gemini-medical-literature
DISEASE_EXTRACTION_PROMPT = """You are an expert pediatric oncologist analyzing patient case notes to identify the primary disease.

Task: Extract the initial diagnosis exactly as written in the case notes.

Examples:
- Input: "A now almost 4-year-old female diagnosed with KMT2A-rearranged AML and CNS2 involvement..."
  Output: AML

- Input: "18 y/o boy, diagnosed in November 2021 with T-ALL with CNS1..."
  Output: T-ALL

- Input: "A 10-year-old patient with relapsed B-cell acute lymphoblastic leukemia (B-ALL)..."
  Output: B-cell acute lymphoblastic leukemia (B-ALL)

Output only the disease name. No additional text or formatting.
"""

# Generic final analysis prompt template
FINAL_ANALYSIS_PROMPT_TEMPLATE = """You are a research analyst synthesizing findings from a comprehensive literature review. Your goal is to provide insights that are valuable for research purposes.

RESEARCH CONTEXT:
Original Query/Case: {case_description}

Primary Focus: {primary_focus}
Key Concepts Searched: {key_concepts}

ANALYZED ARTICLES:
{articles_content}

Based on the research context and analyzed articles above, please provide a comprehensive synthesis in markdown format with the following sections:

## Literature Analysis: {primary_focus}

### 1. Executive Summary
Provide a concise overview of the key findings from the literature review, highlighting:
- Main themes identified across the literature
- Most significant insights relevant to the research query  
- Overall quality and quantity of available evidence
- Key takeaways for researchers in this field

### 2. Key Findings by Concept
| Concept | Articles Discussing | Key Findings | Evidence Quality |
|---------|-------------------|--------------|------------------|
[For each key concept searched, summarize what the literature reveals about it]

### 3. Methodological Landscape
| Research Method | Frequency | Notable Studies | Insights Generated |
|-----------------|-----------|-----------------|-------------------|
[Map the research methodologies used across the analyzed articles]

### 4. Temporal Trends
| Time Period | Research Focus | Key Developments | Paradigm Shifts |
|-------------|----------------|------------------|-----------------|
[Analyze how research in this area has evolved over time]

### 5. Cross-Study Patterns
| Pattern | Supporting Evidence | Implications | Confidence Level |
|---------|-------------------|--------------|------------------|
[Identify patterns that appear across multiple studies]

### 6. Controversies & Unresolved Questions
| Issue | Different Perspectives | Evidence For/Against | Current Consensus |
|-------|----------------------|---------------------|-------------------|
[Highlight areas of disagreement or ongoing debate in the literature]

### 7. Knowledge Gaps & Future Research
| Gap Identified | Why It Matters | Potential Approaches | Expected Impact |
|----------------|----------------|---------------------|-----------------|
[Map areas where further research is needed]

### 8. Practical Applications
Based on the synthesized literature, identify:
- How these findings can be applied in practice
- Recommendations for researchers entering this field
- Tools, methods, or frameworks that emerge from the literature
- Potential interdisciplinary connections

### 9. Quality & Reliability Assessment
Evaluate the overall body of literature:
- **Study Types**: Distribution of research designs (experimental, observational, reviews, etc.)
- **Sample Characteristics**: Common sample sizes, populations studied
- **Geographic Distribution**: Where research is being conducted
- **Publication Patterns**: Journal quality, publication years, citation patterns
- **Methodological Rigor**: Strengths and limitations observed

### 10. Synthesis & Conclusions
Provide an integrated narrative that:
- Connects findings across all analyzed articles
- Identifies the strongest evidence and most reliable findings
- Suggests how this research area is likely to develop
- Offers guidance for stakeholders interested in this topic

### 11. Bibliography
**Most Relevant Articles** (in order of relevance to the research query):
[Format each as: Title, Authors, Journal (Year), [PMID: xxxxx](https://pubmed.ncbi.nlm.nih.gov/xxxxx/)]

---

IMPORTANT NOTES:
- Maintain objectivity and clearly distinguish between strong evidence and preliminary findings
- Use accessible language while preserving scientific accuracy
- All claims must be traceable to specific articles in the analysis
- When evidence is conflicting, present all viewpoints fairly
- Focus on research insights and knowledge synthesis rather than prescriptive recommendations
- Highlight both the strengths and limitations of the current literature
"""

# Events extraction prompt - updated to extract general concepts for better literature matching
EVENT_EXTRACTION_PROMPT = """You are an expert pediatric oncologist analyzing patient case notes to identify key disease concepts and clinical features for literature search.

Task: Extract 5 general medical concepts that would help find relevant literature. Focus on:
- Disease types and subtypes (e.g., "AML", "T-ALL", "B-ALL")
- Genetic alterations (gene names only, e.g., "KMT2A rearrangement", "FLT3 mutation", "TP53 mutation")
- Treatment modalities (e.g., "HSCT", "chemotherapy", "CAR-T therapy", "stem cell transplant")
- General complications (e.g., "relapse", "refractory disease", "CNS involvement", "MRD positive")
- Anatomical sites or disease features (e.g., "bone marrow", "extramedullary disease")

Instructions:
- Extract GENERAL CONCEPTS that appear in medical literature
- DO NOT include patient-specific details like percentages, timeframes, or specific protocol names
- Focus on searchable medical terms
- Output exactly 5 concepts

Example:
Input: "A 4-year-old female with KMT2A-rearranged AML and CNS2 involvement exhibited refractory disease after NOPHO protocol. MRD remained at 35%. She relapsed 10 months after cord blood HSCT with 33% blasts. WES showed KMT2A::MLLT3 fusion and NRAS mutation."

Output: "AML" "KMT2A rearrangement" "CNS involvement" "refractory disease" "HSCT relapse"

Output only 5 general medical concepts, one per line in quotes. No additional text or formatting.
"""

# --- Helper Functions for Enhanced Setup ---
def get_user_credentials_from_gcloud():
    """Get user credentials from gcloud SDK (not application-default)."""
    try:
        import subprocess
        import json
        from google.oauth2 import credentials as oauth2_credentials
        from google.auth.transport.requests import Request
        
        # Get the access token from gcloud
        result = subprocess.run(
            ['gcloud', 'auth', 'print-access-token'],
            capture_output=True,
            text=True,
            check=True
        )
        access_token = result.stdout.strip()
        
        if not access_token:
            return None
        
        # Get account info
        account_result = subprocess.run(
            ['gcloud', 'config', 'get-value', 'account'],
            capture_output=True,
            text=True,
            check=True
        )
        account = account_result.stdout.strip()
        
        # Create credentials from the access token
        # Note: This is a simplified approach - in production you'd want proper token refresh
        credentials = oauth2_credentials.Credentials(
            token=access_token,
            # These scopes should match what gcloud auth provides
            scopes=[
                'https://www.googleapis.com/auth/cloud-platform',
                'https://www.googleapis.com/auth/userinfo.email',
                'openid'
            ]
        )
        
        print(f"✅ Using credentials for account: {account}")
        return credentials
        
    except subprocess.CalledProcessError as e:
        print(f"Error getting gcloud credentials: {e}")
        print("\nPlease authenticate using: gcloud auth login")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

# Global variable to store user credentials
USER_CREDENTIALS = None

def list_projects():
    """List all available Google Cloud projects."""
    global USER_CREDENTIALS
    
    try:
        # Check if credentials are available
        if USER_CREDENTIALS is None:
            print("No user credentials available. Please authenticate first.")
            return []
        
        # Create client with user credentials
        client = resourcemanager_v3.ProjectsClient(credentials=USER_CREDENTIALS)
        projects = []
        request = resourcemanager_v3.SearchProjectsRequest(query="")
        
        for project in client.search_projects(request=request):
            if project.state == resourcemanager_v3.Project.State.ACTIVE:
                projects.append({
                    "id": project.project_id,
                    "name": project.display_name,
                    "number": project.name.split('/')[-1]
                })
        return sorted(projects, key=lambda p: p['id'])
    except Exception as e:
        print(f"Error listing projects: {e}")
        return []

def check_billing_enabled(project_id):
    """Check if billing is enabled for a project."""
    global USER_CREDENTIALS
    try:
        client = billing_v1.CloudBillingClient(credentials=USER_CREDENTIALS)
        billing_info = client.get_project_billing_info(name=f"projects/{project_id}")
        return billing_info.billing_enabled
    except Exception as e:
        print(f"Could not check billing for project {project_id}: {e}")
        return False

def list_enabled_apis(project_id):
    """List enabled APIs for a project."""
    global USER_CREDENTIALS
    try:
        client = service_usage_v1.ServiceUsageClient(credentials=USER_CREDENTIALS)
        request = service_usage_v1.ListServicesRequest(
            parent=f"projects/{project_id}",
            filter="state:ENABLED"
        )
        enabled_apis = [service.name.split('/')[-1] for service in client.list_services(request=request)]
        return enabled_apis
    except Exception as e:
        print(f"Error listing APIs: {e}")
        return []

def enable_apis(project_id, apis_to_enable, progress=gr.Progress()):
    """Enable a list of APIs for a project."""
    global USER_CREDENTIALS
    client = service_usage_v1.ServiceUsageClient(credentials=USER_CREDENTIALS)
    total_apis = len(apis_to_enable)
    for i, api_name in enumerate(apis_to_enable):
        progress((i + 1) / total_apis, desc=f"Enabling {api_name}...")
        try:
            request = service_usage_v1.EnableServiceRequest(name=f"projects/{project_id}/services/{api_name}")
            operation = client.enable_service(request=request)
            operation.result(timeout=300)  # Wait for completion
        except Exception as e:
            raise RuntimeError(f"Error enabling API {api_name}: {e}")
    return True

def list_billing_accounts():
    """Lists available billing accounts and adds an option to create a new one."""
    global USER_CREDENTIALS
    try:
        client = billing_v1.CloudBillingClient(credentials=USER_CREDENTIALS)
        accounts = client.list_billing_accounts()
        account_names = [f"{acc.display_name} ({acc.name.split('/')[-1]})" for acc in accounts if acc.open]
        return account_names + [CREATE_BILLING_ACCOUNT_OPTION]
    except Exception as e:
        print(f"Error listing billing accounts: {e}")
        return [CREATE_BILLING_ACCOUNT_OPTION]

def create_new_project(project_id, billing_account_name, model_endpoint, thinking_budget, progress=gr.Progress()):
    """Creates a new GCP project, links billing, and enables necessary APIs."""
    global USER_CREDENTIALS, MODEL_ID, THINKING_BUDGET
    try:
        # Update global model settings
        MODEL_ID = model_endpoint
        THINKING_BUDGET = thinking_budget
        
        progress(0.1, desc="Creating project...")
        project_client = resourcemanager_v3.ProjectsClient(credentials=USER_CREDENTIALS)
        project = {'project_id': project_id, 'display_name': project_id}
        operation = project_client.create_project(project=project)
        created_project = operation.result(timeout=300)

        progress(0.4, desc="Linking billing account...")
        billing_client = billing_v1.CloudBillingClient(credentials=USER_CREDENTIALS)
        billing_account_id = billing_account_name.split(' ')[-1].strip('()')
        project_billing_info = {'billing_account_name': f"billingAccounts/{billing_account_id}"}
        billing_client.update_project_billing_info(
            name=f"projects/{created_project.project_id}",
            project_billing_info=project_billing_info
        )

        progress(0.6, desc="Enabling APIs...")
        enable_apis(project_id, REQUIRED_APIS, progress)

        # Add a delay to ensure project propagation and IAM permissions
        progress(0.7, desc="Waiting for project propagation...")
        time.sleep(10)  # 10-second delay for IAM permissions to propagate

        # Use the shared setup logic with model endpoint
        global genai_client, bq_client, journal_impact_dict
        genai_client, bq_client, journal_impact_dict = setup_project(project_id, LOCATION, USER_DATASET, model_endpoint, progress)

        return f"✅ Project '{project_id}' created and set up.", f"{project_id} ({project_id})"
    except Exception as e:
        return f"❌ Error creating project: {e}", None

def link_billing_to_project(project_id, billing_account_name):
    """Links an existing billing account to a project."""
    global USER_CREDENTIALS
    try:
        billing_client = billing_v1.CloudBillingClient(credentials=USER_CREDENTIALS)
        billing_account_id = billing_account_name.split(' ')[-1].strip('()')
        project_billing_info = {'billing_account_name': f"billingAccounts/{billing_account_id}"}
        billing_client.update_project_billing_info(
            name=f"projects/{project_id}",
            project_billing_info=project_billing_info
        )
        return True, "✅ Billing account linked successfully!"
    except Exception as e:
        return False, f"❌ Error linking billing account: {e}"

# --- Core Functions ---
def setup_project(project_id, location, dataset, model_endpoint, progress=gr.Progress()):
    """Common setup logic for both new and existing projects."""
    try:
        # Set environment variable
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        
        progress(0.7, desc="Initializing clients...")
        genai_client, bq_client = init_clients(project_id, location)
        if not genai_client or not bq_client:
            raise ConnectionError("Failed to initialize Google Cloud clients.")

        # Setup BigQuery dataset and model with selected model endpoint
        setup_bigquery(project_id, dataset, bq_client, model_endpoint, progress)

        progress(0.9, desc="Loading journal data...")
        journal_impact_dict = load_journal_data(bq_client)
        
        return genai_client, bq_client, journal_impact_dict
    except Exception as e:
        raise e

def init_clients(project_id, location):
    """Initialize clients with retry logic for newly created projects."""
    global USER_CREDENTIALS
    max_retries = 3
    retry_delays = [5, 10, 15]  # Delays in seconds between retries
    
    # Ensure the project ID is set in the environment
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to initialize clients for project {project_id} (attempt {attempt + 1}/{max_retries})...")
            
            # Use user credentials explicitly
            genai_client = genai.Client(vertexai=True, project=project_id, location=location, credentials=USER_CREDENTIALS)
            bq_client = bigquery.Client(project=project_id, credentials=USER_CREDENTIALS)
            
            # Test BigQuery access
            test_query = "SELECT 1"
            bq_client.query(test_query).result()
            
            print(f"Successfully initialized clients for project {project_id}")
            return genai_client, bq_client
            
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                print(f"Waiting {delay} seconds before retry...")
                time.sleep(delay)
            else:
                print(f"All {max_retries} attempts failed. Error initializing clients for project {project_id}: {e}")
                return None, None

def load_journal_data(bq_client_param=None):
    """Load journal data from BigQuery instead of CSV."""
    try:
        # Use provided client or global client
        client = bq_client_param or bq_client
        if client is None:
            print("BigQuery client not initialized")
            return {}
            
        query = f"""
        SELECT 
            journal_title,
            sjr
        FROM `{PROJECT_ID}.{USER_DATASET}.journal_impact`
        WHERE sjr IS NOT NULL
        ORDER BY sjr DESC
        """
        
        print("Loading journal data from BigQuery...")
        start_time = time.time()
        
        results = client.query(query).to_dataframe()
        
        # Convert to dictionary
        sjr_dict = {}
        for _, row in results.iterrows():
            sjr_dict[row['journal_title']] = row['sjr']
        
        load_time = time.time() - start_time
        print(f"✅ Loaded {len(sjr_dict)} journals from BigQuery in {load_time:.2f} seconds")
        return sjr_dict
        
    except Exception as e:
        print(f"Error loading journal data from BigQuery: {e}")
        return {}

def setup_journal_impact_table(bq_client, project_id, dataset_id, progress=gr.Progress()):
    """Create and populate journal impact table if it doesn't exist."""
    table_id = "journal_impact"
    table_ref = f"{project_id}.{dataset_id}.{table_id}"
    
    try:
        # Check if table exists
        try:
            table = bq_client.get_table(table_ref)
            print(f"Journal impact table already exists with {table.num_rows} rows")
            return True
        except:
            # Table doesn't exist, create it
            progress(0.85, desc="Creating journal impact table...")
            print(f"Creating journal impact table: {table_ref}")
            
            # Download and parse CSV
            import pandas as pd
            df = pd.read_csv(JOURNAL_IMPACT_CSV_URL, sep=';')
            
            # Convert SJR values from string with commas to float
            df['SJR_float'] = df['SJR'].apply(lambda x: float(str(x).replace(',', '')) if pd.notna(x) and str(x) != '' else None)
            
            # Select relevant columns and rename
            columns_to_keep = {
                'Title': 'journal_title',
                'SJR_float': 'sjr',
                'Issn': 'issn',
                'SJR Best Quartile': 'sjr_best_quartile',
                'H index': 'h_index',
                'Publisher': 'publisher',
                'Categories': 'categories',
                'Country': 'country',
                'Type': 'type'
            }
            
            df_clean = df[list(columns_to_keep.keys())].rename(columns=columns_to_keep)
            
            # Remove rows with no SJR value
            df_clean = df_clean[df_clean['sjr'].notna()]
            
            print(f"Cleaned data: {len(df_clean)} rows with valid SJR values")
            
            # Define table schema
            schema = [
                bigquery.SchemaField("journal_title", "STRING"),
                bigquery.SchemaField("sjr", "FLOAT64"),
                bigquery.SchemaField("issn", "STRING"),
                bigquery.SchemaField("sjr_best_quartile", "STRING"),
                bigquery.SchemaField("h_index", "INT64"),
                bigquery.SchemaField("publisher", "STRING"),
                bigquery.SchemaField("categories", "STRING"),
                bigquery.SchemaField("country", "STRING"),
                bigquery.SchemaField("type", "STRING"),
            ]
            
            # Configure load job
            job_config = bigquery.LoadJobConfig(
                schema=schema,
                write_disposition="WRITE_TRUNCATE",
            )
            
            # Load data
            print(f"Uploading {len(df_clean)} rows to {table_ref}...")
            job = bq_client.load_table_from_dataframe(df_clean, table_ref, job_config=job_config)
            job.result()  # Wait for job to complete
            
            # Verify upload
            table = bq_client.get_table(table_ref)
            print(f"✅ Successfully created journal impact table with {table.num_rows} rows")
            return True
            
    except Exception as e:
        print(f"Error setting up journal impact table: {e}")
        # Continue without journal impact data
        return False

def extract_medical_info(case_text, client, disease_prompt=None, events_prompt=None):
    """Extract medical information using custom or default prompts."""
    results = {}
    
    # Use provided prompts or defaults
    if not disease_prompt:
        disease_prompt = DISEASE_EXTRACTION_PROMPT
    if not events_prompt:
        events_prompt = EVENT_EXTRACTION_PROMPT
    
    full_disease_prompt = f"{disease_prompt}\n\nCase notes:\n{case_text}"
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[full_disease_prompt], 
        config=GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        )
    )
    results["disease"] = response.text.strip()
    
    full_events_prompt = f"{events_prompt}\n\nCase notes:\n{case_text}"
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[full_events_prompt], 
        config=GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        )
    )
    results["events"] = response.text.strip()
    
    # Generate event IDs for the extracted events
    events_text = results["events"]
    events_list = []
    events_with_ids = {}
    
    # Parse events (handle both line-separated and quote-separated formats)
    if '"' in events_text:
        # Events are in quotes
        import re
        events_list = re.findall(r'"([^"]+)"', events_text)
    else:
        # Events are line-separated
        events_list = [e.strip() for e in events_text.split('\n') if e.strip()]
    
    # Create ID mapping
    for i, event in enumerate(events_list, 1):
        event_id = f"event_{i}"
        events_with_ids[event_id] = event
    
    results["events_list"] = events_list
    results["events_with_ids"] = events_with_ids
    
    return results

def search_pubmed_articles(disease, events, bq_client, embedding_model, pubmed_table, top_k, offset=0):
    query_text = f"{disease} {' '.join(events)}"
    # Debug: print the project being used
    print(f"Using BigQuery project: {bq_client.project}")
    print(f"Using embedding model: {embedding_model}")
    print(f"Searching with top_k={top_k}, offset={offset}")
    
    # Use DECLARE/SET pattern to safely handle special characters in query text
    sql = f"""
    DECLARE query_text STRING;
    SET query_text = \"\"\"
{query_text}
\"\"\";
    
    WITH vector_results AS (
        SELECT base.name AS PMCID, base.PMID, base.content, distance 
        FROM VECTOR_SEARCH(
            TABLE `{pubmed_table}`, 
            'ml_generate_embedding_result', 
            (SELECT ml_generate_embedding_result 
             FROM ML.GENERATE_EMBEDDING(
                 MODEL `{embedding_model}`, 
                 (SELECT query_text AS content)
             )), 
            top_k => {top_k + offset}
        )
    )
    SELECT * FROM vector_results
    ORDER BY distance
    LIMIT {top_k}
    OFFSET {offset}
    """
    
    return bq_client.query(sql).to_dataframe()

def lookup_journal_impact_score(journal_title, journal_dict, genai_client):
    """Look up journal impact score using Gemini API with structured response."""
    if not journal_title or not genai_client:
        return 0
    
    try:
        start_time = time.time()
        
        # Build journal context for Gemini
        journal_context = "\n".join([f"{title}: {sjr}" for title, sjr in journal_dict.items()])
        
        prompt = f"""Given the journal title "{journal_title}", find the matching journal from the list below and return its SJR score.

Journal SJR scores:
{journal_context}

Return the SJR score as an integer. If no match is found or the score is NaN/invalid, return 0."""
        
        # Use structured response configuration
        from google.genai import types
        
        config = types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema={"type": "OBJECT", "properties": {"sjr_score": {"type": "INTEGER"}}},
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        )
        
        response = genai_client.models.generate_content(
            model=MODEL_ID, 
            contents=[prompt], 
            config=config
        )
        
        # Parse JSON response
        try:
            result = json.loads(response.text)
            score = result.get('sjr_score', 0)
            lookup_time = time.time() - start_time
            print(f"   🔍 Gemini journal lookup for '{journal_title}' took {lookup_time:.2f} seconds (found SJR: {score})")
            return score if score >= 0 else 0
        except json.JSONDecodeError:
            lookup_time = time.time() - start_time
            print(f"   🔍 Gemini journal lookup for '{journal_title}' took {lookup_time:.2f} seconds (parse error)")
            return 0
            
    except Exception as e:
        print(f"Error looking up journal impact score: {e}")
        return 0

def calculate_dynamic_score(metadata, criteria_list, journal_dict):
    """Calculate article score based on dynamic criteria configuration."""
    score = 0
    breakdown = {}
    current_year = datetime.now().year
    
    for criterion in criteria_list:
        # Skip if weight is 0
        if criterion['weight'] == 0:
            continue
            
        criterion_type = criterion.get('type', 'boolean')
        criterion_name = criterion['name']
        
        if criterion_type == 'special_journal':
            # Look up journal impact score using Gemini
            journal_title = metadata.get('journal_title', '')
            sjr = lookup_journal_impact_score(journal_title, journal_dict, genai_client)
            
            if sjr > 0:
                # Apply logarithmic scaling with a cap to prevent domination
                # Log scale: log(sjr + 1) * 10, capped at 100
                normalized_sjr = min(math.log(sjr + 1) * 10, 100)
                weighted_score = normalized_sjr * (criterion['weight'] / 100)
                score += weighted_score
                breakdown['journal_impact'] = round(weighted_score, 2)
                
        elif criterion_type == 'special_year':
            # Enhanced year penalty with exponential decay
            year_value = metadata.get('year')
            if year_value is not None and year_value != '':
                try:
                    # Handle both string and int types
                    if isinstance(year_value, str):
                        # Remove any quotes or whitespace
                        year_value = year_value.strip().strip('"').strip("'")
                    article_year = int(year_value)
                    
                    if article_year > 1900 and article_year <= current_year:  # Sanity check
                        year_diff = current_year - article_year
                        # Exponential decay: penalty grows exponentially with age
                        # Base penalty of -10, multiplied by 1.2^year_diff
                        year_penalty = -10 * (1.2 ** min(year_diff, 10))  # Cap at 10 years to prevent overflow
                        # Apply user's weight as a multiplier
                        weighted_penalty = year_penalty * criterion['weight'] / 100  # Normalize weight
                        score += weighted_penalty
                        breakdown['year'] = round(weighted_penalty, 2)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not process year '{year_value}': {e}")
                    pass
                    
        elif criterion_type == 'numeric':
            # For numeric criteria, multiply value by weight
            value = metadata.get(criterion_name, 0)
            if isinstance(value, (int, float)):
                weighted_value = value * criterion['weight']
                score += weighted_value
                breakdown[criterion_name] = round(weighted_value, 2)
                
        elif criterion_type == 'direct':
            # For direct scoring, use the value as-is (ignore weight)
            value = metadata.get(criterion_name, 0)
            if isinstance(value, (int, float)):
                score += value
                breakdown[criterion_name] = round(value, 2)
                
        else:
            # Default: boolean criteria
            if metadata.get(criterion_name):
                score += criterion['weight']
                breakdown[criterion_name] = criterion['weight']
                
    return round(score, 2), breakdown

def analyze_event_coverage_batch(df, disease, events_with_ids, bq_client):
    """Phase 1: Analyze articles for event coverage using AI.GENERATE_TABLE."""
    global PROJECT_ID, USER_DATASET, PUBMED_TABLE
    
    if df.empty:
        return []
    
    try:
        print(f"\n📊 Phase 1: Checking event coverage for {len(df)} article(s)...")
        coverage_start_time = time.time()
        
        # Build event list with IDs for the prompt
        event_list_text = "\n".join([f"- {event_id}: \"{event_text}\"" 
                                     for event_id, event_text in events_with_ids.items()])
        
        # Get PMCIDs from dataframe (using name field as primary identifier)
        # Handle cases where PMCID might be None
        pmcids = [str(pmcid) for pmcid in df['PMCID'].tolist() if pmcid is not None]
        if not pmcids:
            print("Warning: No valid PMCIDs found in batch")
            return []
        pmcids_str = "', '".join(pmcids)
        
        # Build PMCID to name mapping for prompt
        pmcid_mapping = {}
        for _, row in df.iterrows():
            pmcid_mapping[str(row['PMCID'])] = str(row['PMCID'])
        
        # Simple prompt focused only on event detection
        coverage_prompt = f"""You are analyzing medical literature to identify which events from a patient case are mentioned in each article.

Master events from patient case:
{event_list_text}

IMPORTANT: You must return JSON with one field:
- event_ids: Comma-separated event IDs found (e.g., "event_1,event_3"), or empty string if no events found

Article PMCID: {{PMCID}}
Article content:
"""
        
        # Escape triple quotes if needed
        coverage_prompt_escaped = coverage_prompt.replace('"""', '\\"""')
        
        # Schema only includes event_ids (name will be passed through automatically)
        schema = "event_ids STRING"
        
        # Construct AI.GENERATE_TABLE query for batch processing using PMCID
        query = f'''
        SELECT 
            name AS PMCID,
            event_ids
        FROM 
        AI.GENERATE_TABLE(
            MODEL `{PROJECT_ID}.{USER_DATASET}.gemini_generation`,
            (
                SELECT 
                    name,
                    CONCAT(
                        REPLACE("""{coverage_prompt_escaped}""", '{{PMCID}}', name),
                        content
                    ) AS prompt
                FROM `{PUBMED_TABLE}`
                WHERE name IN ('{pmcids_str}')
            ),
            STRUCT(
                "{schema}" AS output_schema,
                4096 AS max_output_tokens,
                0 AS temperature,
                0.95 AS top_p
            )
        )
        '''
        
        # Execute query
        results_df = bq_client.query(query).to_dataframe()
        
        # Convert to list of dictionaries
        results = []
        for _, row in results_df.iterrows():
            result = {
                'PMCID': row['PMCID'],
                'event_ids': row.get('event_ids', '').strip()
            }
            results.append(result)
        
        coverage_time = time.time() - coverage_start_time
        print(f"   ✅ Event coverage check completed in {coverage_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error in event coverage analysis: {str(e)}")
        return []

def build_dynamic_schema(criteria):
    """Build dynamic BigQuery schema based on criteria configuration."""
    # Start with standard fields
    schema_parts = [
        "title STRING",
        "journal_title STRING", 
        "year STRING", 
        "paper_type STRING",
        "actionable_events STRING"
    ]
    
    # Add fields for each criterion based on type
    for criterion in criteria:
        if criterion['name'] not in ['journal_impact', 'year']:  # Skip special ones already handled
            if criterion['type'] == 'boolean':
                schema_parts.append(f"{criterion['name']} BOOL")
            elif criterion['type'] in ['numeric', 'direct']:
                schema_parts.append(f"{criterion['name']} INT64")
    
    return ",\n    ".join(schema_parts)

def analyze_article_batch_with_criteria(df, disease, events, bq_client, journal_dict, persona, criteria):
    """Analyze articles using AI.GENERATE_TABLE directly on BigQuery table."""
    global PROJECT_ID, USER_DATASET, PUBMED_TABLE
    
    if df.empty:
        return []
    
    try:
        print(f"\n📊 Starting AI.GENERATE_TABLE analysis for {len(df)} article(s)...")
        ai_start_time = time.time()
        # Build criteria instructions
        criteria_instructions = []
        for criterion in criteria:
            if criterion['name'] not in ['journal_impact', 'year']:
                if criterion['type'] == 'boolean':
                    criteria_instructions.append(f"- {criterion['name']} (boolean): {criterion['description']}")
                elif criterion['type'] == 'numeric':
                    criteria_instructions.append(f"- {criterion['name']} (number): {criterion['description']} (Return 0 if unknown)")
                elif criterion['type'] == 'direct':
                    criteria_instructions.append(f"- {criterion['name']} (number 0-100): {criterion['description']} (Return 0 if no matches or unknown)")
        
        criteria_text = "\n".join(criteria_instructions) if criteria_instructions else ""
        
        # Build dynamic schema
        schema = build_dynamic_schema(criteria)
        
        # Build the complete prompt in Python first
        full_prompt = f"""{persona}

Analyze this article for relevance to:
Disease: {disease}
Events: {', '.join(events)}

For each article, extract the following information:
1. Standard fields (always extract these):
   - title: Article title (if unknown, return empty string)
   - journal_title: Name of the journal (if unknown, return empty string)
   - year: Publication year as a string (e.g., "2023"). If unknown or not found, return empty string, NOT null or NaN
   - paper_type: Type of paper (e.g., clinical trial, review, case report)
   - actionable_events: Comma-separated list of events found in the article

2. Evaluation criteria:
{criteria_text}

IMPORTANT: For all numeric fields, always return 0 instead of null, NaN, or leaving the field empty.

Article content:
"""
        
        # Escape triple quotes if they appear in the prompt (unlikely but safe)
        full_prompt_escaped = full_prompt.replace('"""', '\\"""')
        
        # Get PMCIDs from dataframe (using name field as primary identifier)
        # Handle cases where PMCID might be None
        pmcids = [str(pmcid) for pmcid in df['PMCID'].tolist() if pmcid is not None]
        if not pmcids:
            print("Warning: No valid PMCIDs found in batch for full analysis")
            return []
        pmcids_str = "', '".join(pmcids)
        
        # Format schema for single line
        schema_single_line = schema.replace('\n', ' ').replace('    ', '')
        
        # Construct AI.GENERATE_TABLE query using PMCID as primary identifier
        query = f'''
        SELECT 
            PMCID,
            PMID,
            * EXCEPT (PMCID, PMID, prompt, full_response, status)
        FROM 
        AI.GENERATE_TABLE(
            MODEL `{PROJECT_ID}.{USER_DATASET}.gemini_generation`,
            (
                SELECT 
                    name AS PMCID,
                    PMID,
                    CONCAT(
                        """{full_prompt_escaped}""",
                        content
                    ) AS prompt
                FROM `{PUBMED_TABLE}`
                WHERE name IN ('{pmcids_str}')
            ),
            STRUCT(
                """{schema_single_line}""" AS output_schema,
                8192 AS max_output_tokens,
                0 AS temperature,
                0.95 AS top_p
            )
        )
        '''
        
        # Execute query
        query_execution_start = time.time()
        results_df = bq_client.query(query).to_dataframe()
        query_execution_time = time.time() - query_execution_start
        print(f"   ⚡ AI.GENERATE_TABLE query executed in {query_execution_time:.2f} seconds")
        
        # Convert to list of dictionaries and preserve article content
        processing_start = time.time()
        results = []
        for _, result_row in results_df.iterrows():
            result_dict = result_row.to_dict()
            
            # Clean up year field if it exists
            if 'year' in result_dict:
                year_val = result_dict['year']
                if year_val in [None, 'NaN', 'nan', 'null', '']:
                    result_dict['year'] = ''
                elif isinstance(year_val, str):
                    # Clean the year string
                    result_dict['year'] = year_val.strip()
            
            # Clean up all INT64 fields (numeric and direct type criteria)
            for criterion in criteria:
                if criterion['type'] in ['numeric', 'direct'] and criterion['name'] in result_dict:
                    field_value = result_dict[criterion['name']]
                    # Handle NaN, null, or invalid values
                    if pd.isna(field_value) or field_value in [None, 'NaN', 'nan', 'null', '']:
                        result_dict[criterion['name']] = 0
                    else:
                        try:
                            # Try to convert to int, default to 0 if it fails
                            result_dict[criterion['name']] = int(float(str(field_value)))
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert {criterion['name']} value '{field_value}' to int, defaulting to 0")
                            result_dict[criterion['name']] = 0
            
            # Find the corresponding content from the original df using PMCID
            matching_row = df[df['PMCID'] == result_dict.get('PMCID')]
            if not matching_row.empty:
                result_dict['content'] = matching_row.iloc[0]['content']
                # Keep PMID if available for PubMed links
                if 'PMID' in matching_row.columns:
                    result_dict['PMID'] = matching_row.iloc[0].get('PMID')
            results.append(result_dict)
        
        total_ai_time = time.time() - ai_start_time
        print(f"   ✅ Total AI.GENERATE_TABLE analysis took {total_ai_time:.2f} seconds")
        
        return results
        
    except Exception as e:
        print(f"Error in AI.GENERATE_TABLE analysis: {str(e)}")
        return []

def estimate_tokens(text):
    """Rough estimate of token count for a text."""
    # Approximation: 1 token ≈ 4 characters
    return len(text) // 4

def format_article_for_analysis(article, idx):
    """Format a single article for the analysis prompt."""
    metadata = article.get('metadata', article)
    
    # Get events found
    events_found = metadata.get('actionable_events', 'None')
    if isinstance(events_found, str) and events_found:
        events_str = events_found
    else:
        events_str = "None identified"
    
    return f"""
Article {idx}:
Title: {metadata.get('title', 'Unknown')}
Journal: {metadata.get('journal_title', 'Unknown')} | Year: {metadata.get('year', 'N/A')}
Type: {metadata.get('paper_type', 'Unknown')}
Score: {article.get('score', 0):.1f}
Key Concepts Found: {events_str}
PMID: {article.get('pmid', 'N/A')} | PMCID: {article.get('pmcid', 'N/A')}

Full Text:
{article.get('content', 'No content available')}
"""

def create_final_analysis_prompt(case_text, disease, events, selected_articles):
    """Create the final analysis prompt with full article contents."""
    
    if not selected_articles:
        return None
    
    # Sort articles by score
    sorted_articles = sorted(selected_articles, key=lambda x: x.get('score', 0), reverse=True)
    
    # Format all selected articles
    articles_content_parts = []
    for idx, article in enumerate(sorted_articles, 1):
        articles_content_parts.append(format_article_for_analysis(article, idx))
    
    # Join all articles with separator
    articles_content = ("\n" + "="*80 + "\n").join(articles_content_parts)
    
    # Fill in the template
    filled_prompt = FINAL_ANALYSIS_PROMPT_TEMPLATE.format(
        case_description=case_text,
        primary_focus=disease,
        key_concepts=', '.join(events),
        articles_content=articles_content
    )
    
    return filled_prompt

def generate_final_analysis_stream(case_text, disease, events, selected_articles, genai_client):
    """Generate final analysis with streaming response."""
    
    if not selected_articles:
        yield "❌ No articles selected. Please select at least one article for analysis."
        return
    
    if not genai_client:
        yield "❌ Gemini client not initialized. Please complete setup first."
        return
    
    try:
        # Create the prompt
        prompt = create_final_analysis_prompt(case_text, disease, events, selected_articles)
        
        if not prompt:
            yield "❌ Could not create analysis prompt."
            return
        
        # Generate content with streaming
        config = GenerateContentConfig(
            temperature=0.3,  # Slightly higher for more creative synthesis
            max_output_tokens=8192,
            candidate_count=1,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        )
        
        # First yield to show we're starting
        yield "🔄 Generating comprehensive analysis...\n\n"
        
        # Stream the response
        try:
            response_stream = genai_client.models.generate_content_stream(
                model=MODEL_ID,
                contents=[prompt],
                config=config
            )
            
            # Accumulate the full response for streaming
            accumulated_text = ""
            
            for chunk in response_stream:
                if hasattr(chunk, 'candidates') and chunk.candidates:
                    for candidate in chunk.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                if hasattr(part, 'text'):
                                    accumulated_text += part.text
                                    # Yield the accumulated content so far
                                    yield accumulated_text
            
            # If no content was generated
            if not accumulated_text:
                yield "❌ No analysis was generated. Please try again."
                
        except Exception as e:
            yield f"❌ Error during analysis generation: {str(e)}"
        
    except Exception as e:
        yield f"❌ Error generating final analysis: {str(e)}"

def setup_bigquery(project, dataset, client, model_endpoint, progress=gr.Progress()):
    """Setup BigQuery dataset and models with retry logic."""
    progress(0.8, desc="Setting up BigQuery dataset and models (may take a couple minutes if first time)...")
    
    # Create dataset if it doesn't exist
    try:
        client.get_dataset(f"{project}.{dataset}")
    except:
        client.create_dataset(bigquery.Dataset(f"{project}.{dataset}"), exists_ok=True)
    
    # Setup journal impact table
    setup_journal_impact_table(client, project, dataset, progress)
    
    # Create text embedding model
    embed_model_query = f"CREATE MODEL IF NOT EXISTS `{project}.{dataset}.textembed` REMOTE WITH CONNECTION DEFAULT OPTIONS(endpoint='text-embedding-005');"
    
    # Create Gemini generation model with dynamic model endpoint
    gen_model_query = f"CREATE OR REPLACE MODEL `{project}.{dataset}.gemini_generation` REMOTE WITH CONNECTION DEFAULT OPTIONS(endpoint='{model_endpoint}');"
    
    models_to_create = [
        ("text embedding", embed_model_query),
        ("Gemini generation", gen_model_query)
    ]
    
    max_retries = 3
    retry_delays = [5, 10, 15]
    
    for model_name, model_query in models_to_create:
        for attempt in range(max_retries):
            try:
                print(f"Creating BigQuery {model_name} model (attempt {attempt + 1}/{max_retries})...")
                client.query(model_query).result()
                print(f"Successfully created {model_name} model for {project}.{dataset}")
                break
                
            except Exception as e:
                error_msg = str(e)
                print(f"Attempt {attempt + 1} failed for {model_name}: {error_msg}")
                
                # Check if it's a job execution error that might be timing-related
                if "internal error during execution" in error_msg.lower() and attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    print(f"This appears to be a timing issue. Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                elif attempt < max_retries - 1:
                    # For other errors, also retry but with shorter delay
                    delay = retry_delays[attempt] // 2
                    print(f"Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for {model_name}.")
                    raise Exception(f"Failed to create {model_name} model after {max_retries} attempts. Last error: {error_msg}")
    
    return f"✅ BigQuery setup complete for {project}.{dataset}"

# --- Gradio App Logic ---
def get_initial_projects():
    """Get the list of projects for the dropdown."""
    projects = list_projects()
    if not projects:
        # Provide option to manually enter project ID
        return gr.update(choices=["[Enter Project ID Manually]"], value="[Enter Project ID Manually]"), "⚠️ Could not list projects automatically. You can either fix the authentication issue (see console) or enter your project ID manually in the field below."
    choices = [f"{p['name']} ({p['id']})" for p in projects]
    return gr.update(choices=choices, value=choices[0] if choices else None), f"✅ Found {len(projects)} projects. Select a project and click Proceed."

def proceed_with_project(project_selection, model_selection, thinking_budget, progress=gr.Progress()):
    """Check and set up the selected project, then move to the next tab."""
    global genai_client, bq_client, journal_impact_dict, PROJECT_ID, LOCATION, USER_DATASET, MODEL_ID, THINKING_BUDGET
    if not project_selection:
        return "❌ Please select a project first.", gr.update(interactive=False), gr.update()

    project_id = project_selection.split('(')[-1].rstrip(')')
    PROJECT_ID = project_id
    
    # Update global model settings
    MODEL_ID = MODEL_OPTIONS[model_selection]
    THINKING_BUDGET = thinking_budget
    
    # Clear any existing project environment variable
    if 'GOOGLE_CLOUD_PROJECT' in os.environ:
        del os.environ['GOOGLE_CLOUD_PROJECT']
    
    # Set the new project ID
    os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

    try:
        progress(0.1, desc="Checking billing status...")
        if not check_billing_enabled(project_id):
            # Return special status to trigger billing setup
            return "billing_needed", gr.update(interactive=False), gr.update()

        progress(0.2, desc="Checking required APIs...")
        enabled_apis = list_enabled_apis(project_id)
        missing_apis = [api for api in REQUIRED_APIS if api not in enabled_apis]
        if missing_apis:
            enable_apis(project_id, missing_apis, progress)

        # Use the shared setup logic with model endpoint
        genai_client, bq_client, journal_impact_dict = setup_project(PROJECT_ID, LOCATION, USER_DATASET, MODEL_ID, progress)

        status = f"✅ Setup complete for {PROJECT_ID} with model {model_selection}! You can now analyze a case."
        return status, gr.update(interactive=True), gr.update(selected=2)

    except Exception as e:
        return f"❌ Error: {e}", gr.update(interactive=False), gr.update()

def create_app(share=False):
    """Create and return the Gradio app."""
    css = """
    .gradio-container { font-family: 'Google Sans', sans-serif; }
    label, .label-wrap, .gradio-label { 
        background-color: transparent !important; 
        border: none !important; 
        box-shadow: none !important; 
        padding: 0 !important; 
    }
    .label-wrap {
        border: none !important;
    }
    .type-info {
        font-size: 0.9em;
        color: #666;
    }
    .extraction-result {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        font-family: monospace;
        white-space: pre-wrap;
        color: #212529 !important;  /* Ensure dark text */
    }
    .extraction-result * {
        color: #212529 !important;  /* Ensure all child elements have dark text */
    }
    .article-card {
        margin: 20px 0;
        padding: 20px;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        background-color: #e8f4f8;  /* Light blue background */
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .score-breakdown-box {
        margin: 15px 0;
        padding: 15px;
        background-color: #d1ecf1;  /* Slightly darker blue */
        border-radius: 5px;
        border: 1px solid #bee5eb;
    }
    .article-content-box {
        margin-top: 10px;
        padding: 15px;
        background-color: #ffffff;  /* White background for article text */
        border-radius: 5px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #dee2e6;
    }
    .article-content-box pre {
        white-space: pre-wrap;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        margin: 0;
        font-size: 14px;
        line-height: 1.6;
        color: #212529 !important;  /* Black text for readability */
    }
    .final-analysis-display {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 30px;
        margin: 20px 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.8;
        color: #212529 !important;
        max-height: 800px;
        overflow-y: auto;
    }
    .final-analysis-display * {
        color: #212529 !important;
    }
    .final-analysis-display h2 {
        color: #0066cc !important;
        border-bottom: 2px solid #0066cc;
        padding-bottom: 10px;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .final-analysis-display h3 {
        color: #333 !important;
        margin-top: 25px;
        margin-bottom: 15px;
    }
    .final-analysis-display table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        background-color: #fff !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .final-analysis-display th {
        background-color: #f8f9fa !important;
        border: 1px solid #dee2e6;
        padding: 12px;
        text-align: left;
        font-weight: bold;
        color: #212529 !important;
    }
    .final-analysis-display td {
        border: 1px solid #dee2e6;
        padding: 12px;
        vertical-align: top;
        color: #212529 !important;
        word-wrap: break-word;
        word-break: break-word;
        max-width: 300px;
    }
    .final-analysis-display td * {
        color: #212529 !important;
    }
    .final-analysis-display a {
        color: #0066cc !important;
        text-decoration: none;
    }
    .final-analysis-display a:hover {
        text-decoration: underline;
    }
    .final-analysis-display ul, .final-analysis-display ol {
        margin: 10px 0;
        padding-left: 30px;
        color: #212529 !important;
    }
    .final-analysis-display li {
        margin: 5px 0;
        color: #212529 !important;
    }
    .final-analysis-display p {
        color: #212529 !important;
    }
    .final-analysis-display span {
        color: #212529 !important;
    }
    """
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"), css=css) as demo:
        gr.Markdown("# 🏥 PubMed Literature Analysis")
        app_state = gr.State({})

        with gr.Tabs() as tabs:
            with gr.TabItem("Get Started", id=0):
                gr.Markdown("## Welcome to the PubMed Literature Analysis Tool")
                gr.Markdown("This tool helps you analyze medical cases using PubMed literature with BigQuery vector search and Gemini. Get started by setting up your Google Cloud project.")
                gr.Markdown("""⚠️ **Important Notice: Demonstration Tool**\n\nThis PubMed literature analysis tool is a **DEMONSTRATION** showcasing AI-powered research capabilities.\n\n- For **research and educational purposes only**\n- **NOT** intended for treatment planning or clinical decisions\n- All AI-generated analyses should be verified against primary sources\n- Results may contain inaccuracies or limitations\n- Users are responsible for appropriate use within research contexts\n\nBy proceeding, you acknowledge these limitations and agree to use this tool responsibly for research purposes only.\n""")
                
                # Authentication Section
                with gr.Column() as auth_section:
                    gr.Markdown("### 🔐 First, authenticate with Google Cloud")
                    gr.Markdown("""
                    **Before using this app, you need to authenticate with Google Cloud:**
                    
                    1. Open a terminal and run:
                       ```
                       gcloud auth login
                       ```
                    
                    2. Follow the prompts to sign in with your Google account
                    
                    3. Once authenticated, click the button below to continue
                    """)
                
                start_button = gr.Button("Get Started", variant="primary")

            with gr.TabItem("1. Setup", id=1):
                status_output = gr.Markdown(value="Loading projects...")
                with gr.Row():
                    project_dropdown = gr.Dropdown(label="Select Google Cloud Project", interactive=True)
                    create_project_btn = gr.Button("Create New Project")

                with gr.Column(visible=False) as create_project_box:
                    gr.Markdown("### Create New Google Cloud Project")
                    new_project_id_input = gr.Textbox(label="New Project ID", placeholder="e.g., pubmed-analysis-123")
                    billing_account_dropdown = gr.Dropdown(label="Select Billing Account")
                    billing_link_message = gr.Markdown(visible=False)
                    create_project_submit_btn = gr.Button("Create and Select Project", variant="primary")
                    cancel_create_project_btn = gr.Button("Cancel")

                with gr.Column(visible=False) as billing_setup_box:
                    gr.Markdown("### 💳 Billing Setup Required")
                    gr.Markdown("This project needs a billing account to use Google Cloud services.")
                    billing_setup_dropdown = gr.Dropdown(label="Select Billing Account")
                    billing_setup_message = gr.Markdown(visible=False)
                    link_billing_btn = gr.Button("Link Billing Account", variant="primary")
                    billing_status = gr.Markdown()

                with gr.Column() as setup_details_box:
                    # Model Configuration Section
                    gr.Markdown("### 🤖 Model Configuration")
                    
                    with gr.Row():
                        model_dropdown = gr.Dropdown(
                            label="Select Model",
                            choices=list(MODEL_OPTIONS.keys()),
                            value="Gemini 2.5 Flash (Default)",
                            interactive=True,
                            info="Choose the Gemini model to use for analysis"
                        )
                    
                    with gr.Row():
                        thinking_budget_slider = gr.Slider(
                            label="Thinking Budget",
                            minimum=0,
                            maximum=24576,
                            value=0,
                            step=1,
                            interactive=True,
                            info="Set the thinking budget for model responses (0 = default)"
                        )
                    
                    gr.Markdown("---")
                    proceed_btn = gr.Button("Proceed", variant="primary")

            with gr.TabItem("2. Case", id=2):
                # Add header row with example button
                with gr.Row():
                    gr.Markdown("## Patient Case Notes")
                    load_example_btn = gr.Button("Load Example", size="sm", scale=0)
                
                # Empty case input by default
                case_input = gr.Textbox(
                    label="", 
                    value="",  # Empty instead of SAMPLE_CASE
                    lines=10,
                    placeholder="Enter patient case details here..."
                )
                
                # Extract Information section
                gr.Markdown("---")
                
                with gr.Column():
                    # Info text inline
                    gr.Markdown("#### Extract Information <span style='color: #666; font-size: 0.85em; font-weight: normal; margin-left: 10px;'>This will help the BigQuery vector search be more refined</span>")
                    
                    # Prompt input fields
                    disease_prompt_input = gr.Textbox(
                        label="Disease Extraction Prompt",
                        value="",
                        lines=5,
                        placeholder="Enter prompt for disease extraction..."
                    )
                    
                    events_prompt_input = gr.Textbox(
                        label="Events Extraction Prompt", 
                        value="",
                        lines=5,
                        placeholder="Enter prompt for actionable events extraction..."
                    )
                    
                    # Extract button
                    extract_btn = gr.Button("Extract", variant="secondary")
                    
                    # Loading indicator
                    extraction_loading = gr.Markdown("🔄 Extracting information...", visible=False)
                    
                    # Extraction results box
                    with gr.Column(visible=False) as extraction_box:
                        gr.Markdown("### Extracted Information")
                        gr.Markdown("*You can edit the extracted values below before proceeding with the analysis.*")
                        
                        # Editable fields
                        with gr.Row():
                            disease_edit = gr.Textbox(
                                label="Disease",
                                value="",
                                lines=1,
                                interactive=True,
                                elem_id="disease_edit"
                            )
                            reset_disease_btn = gr.Button("Reset", size="sm", scale=0)
                        
                        events_edit = gr.Textbox(
                            label="Actionable Events",
                            value="",
                            lines=3,
                            interactive=True,
                            elem_id="events_edit",
                            info="One event per line or comma-separated"
                        )
                        reset_events_btn = gr.Button("Reset to AI suggestion", size="sm")
                        
                        # Original AI extraction display (for reference)
                        with gr.Accordion("View Original AI Extraction", open=False):
                            ai_extraction_display = gr.Markdown("")
                    
                    # Store extraction state
                    extraction_state = gr.State({"extracted": False, "disease": "", "events": ""})
                
                # Hidden slider - keeps default value of 10
                num_articles_slider = gr.Slider(
                    5, 50, 10, 
                    step=1, 
                    label="Number of Articles to Analyze",
                    visible=False  # Hide the slider
                )
                
                # Changed button text and purpose
                proceed_to_persona_btn = gr.Button("Proceed", variant="primary", interactive=False)
                case_status = gr.Markdown()

            with gr.TabItem("3. Persona", id=3):
                gr.Markdown("## Customize Your Analysis Persona")
                gr.Markdown("*Define your research perspective and customize how articles will be scored for relevance.*")
                
                # Persona Section (Top Box)
                with gr.Column():
                    gr.Markdown("### Analysis Persona")
                    with gr.Row():
                        with gr.Column(scale=4):
                            persona_text = gr.Textbox(
                                label="",
                                value="You are a medical researcher analyzing literature for clinical relevance and treatment insights.",
                                lines=4,
                                placeholder="Describe your research perspective and goals..."
                            )
                        with gr.Column(scale=1):
                            load_persona_btn = gr.Button("Load Example", size="sm")
                            
                    # Example personas (hidden, for dropdown)
                    example_personas = gr.State({
                        "Clinical Researcher": "You are a pediatric oncologist focused on finding the latest treatment protocols and clinical trial results for childhood cancers. Prioritize evidence-based therapies with proven efficacy.",
                        "Pharmaceutical Developer": "You are a pharmaceutical researcher looking for novel drug targets and biomarkers with strong preclinical and clinical evidence. Focus on mechanistic insights and translational potential.",
                        "Patient Advocate": "You are evaluating treatment options from a patient perspective, prioritizing safety profiles, quality of life outcomes, and accessibility of treatments.",
                        "Basic Scientist": "You are a molecular biologist interested in understanding disease mechanisms at the cellular and molecular level. Focus on novel pathways, genetic factors, and potential therapeutic targets."
                    })
                
                gr.Markdown("---")
                
                # Scoring Criteria Section (Bottom Box)
                with gr.Column():
                    gr.Markdown("### Article Scoring Criteria")
                    gr.Markdown("*Adjust weights to prioritize what matters most for your analysis. Articles will be scored based on these criteria.*")
                    
                    # We'll store criteria in state
                    criteria_state = gr.State([])
                    
                    # Add controls
                    with gr.Row():
                        add_criterion_btn = gr.Button("➕ Add New Criterion", size="sm")
                        reset_criteria_btn = gr.Button("🔄 Reset to Defaults", size="sm")
                        
                    # Total weight display
                    total_weight_display = gr.Markdown("**Total Weight:** 0")
                    
                    # Container for criteria - pre-create 20 slots
                    criterion_rows = []
                    criterion_descriptions = []
                    criterion_type_infos = []
                    criterion_sliders = []
                    criterion_delete_btns = []
                    
                    with gr.Column() as criteria_container:
                        for i in range(20):  # Create 20 slots
                            with gr.Row(visible=False) as row:
                                with gr.Column(scale=3):
                                    desc = gr.Markdown("")
                                    type_info = gr.Markdown("", elem_classes="type-info")
                                
                                with gr.Column(scale=2):
                                    slider = gr.Slider(
                                        minimum=0,
                                        maximum=100,
                                        value=0,
                                        label="Weight",
                                        step=1
                                    )
                                
                                with gr.Column(scale=1):
                                    delete_btn = gr.Button("🗑️ Delete", variant="stop", size="sm")
                            
                            criterion_rows.append(row)
                            criterion_descriptions.append(desc)
                            criterion_type_infos.append(type_info)
                            criterion_sliders.append(slider)
                            criterion_delete_btns.append(delete_btn)
                    
                    # Add criterion dialog
                    with gr.Column(visible=False) as add_criterion_dialog:
                        gr.Markdown("### Add New Scoring Criterion")
                        new_criterion_description = gr.Textbox(
                            label="Description",
                            placeholder="e.g., Does the article include safety data?",
                            lines=2
                        )
                        new_criterion_type = gr.Dropdown(
                            label="Type",
                            choices=["boolean", "numeric", "direct"],
                            value="boolean",
                            info="Boolean: Yes/No criteria | Numeric: Multiplies value by weight | Direct: Uses value as-is"
                        )
                        new_criterion_weight = gr.Slider(
                            label="Weight",
                            minimum=0,
                            maximum=100,
                            value=10,
                            step=1
                        )
                        with gr.Row():
                            confirm_add_btn = gr.Button("Add", variant="primary", size="sm")
                            cancel_add_btn = gr.Button("Cancel", size="sm")
                    
                analyze_btn = gr.Button("Retrieve and analyze articles", variant="primary", interactive=False)
                analysis_status = gr.Markdown()

            with gr.TabItem("4. Results", id=4):
                # Analysis configuration section
                with gr.Column():
                    gr.Markdown("### Analysis Configuration")
                    with gr.Row():
                        default_articles_input = gr.Number(
                            label="Default Articles to Retrieve",
                            value=5,
                            minimum=1,
                            maximum=50,
                            step=1,
                            info="Initial number of articles to retrieve from PubMed"
                        )
                        min_articles_per_event = gr.Number(
                            label="Minimum Articles per Event",
                            value=3,
                            minimum=1,
                            maximum=10,
                            step=1,
                            info="Minimum number of articles required for each actionable event"
                        )
                    start_analysis_btn = gr.Button("Start Analysis", variant="primary")
                    
                    # Configuration state
                    analysis_config = gr.State({
                        "default_articles": 5,
                        "min_per_event": 3,
                        "max_articles": 50
                    })
                
                gr.Markdown("---")
                
                # Progress tracking components
                with gr.Row():
                    analysis_progress = gr.Markdown("Configure settings above and click 'Start Analysis' to begin.")
                    stop_analysis_btn = gr.Button("Stop Analysis", variant="stop", visible=False)
                
                # Event coverage display
                event_coverage_display = gr.Markdown(visible=False)
                
                # Live results display - hidden as it's redundant
                live_results_df = gr.DataFrame(
                    label="Article Analysis Results (Live)",
                    headers=["Score", "Title", "Journal", "Year", "Status"],
                    interactive=False,
                    visible=False  # Hide the redundant table
                )
                
                # Detailed analysis display - expanded by default
                with gr.Accordion("Detailed Analysis", open=True) as analysis_accordion:
                    detailed_analysis_html = gr.HTML()
                
                # State to track results
                results_state = gr.State([])
                analysis_active = gr.State(False)
                
                # Final Analysis Button - appears after analysis (replaces summary)
                with gr.Row():
                    # Empty column for centering
                    gr.Column()
                    with gr.Column(scale=2):
                        final_analysis_btn = gr.Button("Proceed to Final Analysis", variant="primary", visible=False)
                    gr.Column()

            with gr.TabItem("5. Final Analysis", id=5):
                gr.Markdown("## Final Analysis")
                gr.Markdown("*Select articles to include in the comprehensive synthesis, then generate the final analysis.*")
                
                # Article selection section
                with gr.Column():
                    gr.Markdown("### Select Articles for Analysis")
                    
                    # Selection controls
                    with gr.Row():
                        select_all_btn = gr.Button("Select All", size="sm")
                        deselect_all_btn = gr.Button("Deselect All", size="sm")
                        selected_count = gr.Markdown("0 articles selected")
                        token_estimate = gr.Markdown("Estimated tokens: 0")
                    
                    # Article checkboxes (will be populated dynamically)
                    article_selections = gr.CheckboxGroup(
                        label="",
                        choices=[],
                        value=[],
                        elem_id="article_selections"
                    )
                
                gr.Markdown("---")
                
                # Analysis Prompt Section
                with gr.Column():
                    with gr.Accordion("📝 Customize Analysis Prompt", open=False):
                        gr.Markdown("*Edit the prompt template below to customize how the final analysis is generated. Use the placeholders `{case_description}`, `{primary_focus}`, `{key_concepts}`, and `{articles_content}` to include dynamic content.*")
                        
                        analysis_prompt_input = gr.Textbox(
                            label="Analysis Prompt Template",
                            value=FINAL_ANALYSIS_PROMPT_TEMPLATE,
                            lines=20,
                            placeholder="Enter your custom analysis prompt template...",
                            elem_id="analysis_prompt_input"
                        )
                        
                        with gr.Row():
                            reset_prompt_btn = gr.Button("Reset to Default", size="sm")
                            load_example_prompt_btn = gr.Button("Load Example", size="sm")
                    
                    # Generate button
                    generate_final_btn = gr.Button(
                        "Generate Final Analysis", 
                        variant="primary", 
                        interactive=False
                    )
                
                gr.Markdown("---")
                
                # Analysis display section
                with gr.Column():
                    final_analysis_progress = gr.Markdown("")
                    final_analysis_display = gr.Markdown(
                        "", 
                        elem_classes="final-analysis-display"
                    )
                
                # State to track selected articles
                selected_articles_state = gr.State([])
                
                # Example prompts state
                example_prompts = gr.State({
                    "Clinical Focus": """You are a clinical researcher synthesizing findings for medical practice.

CASE: {case_description}
DISEASE: {primary_focus}
KEY FACTORS: {key_concepts}

ANALYZED ARTICLES:
{articles_content}

Please provide a clinical synthesis with these sections:

## Clinical Summary: {primary_focus}

### Key Clinical Findings
- Summarize the most clinically relevant findings
- Focus on treatment efficacy and safety
- Highlight patient outcomes

### Treatment Recommendations
| Treatment | Evidence Level | Success Rate | Side Effects |
|-----------|---------------|--------------|--------------|
[Analyze treatments from the articles]

### Clinical Pearls
- List key takeaways for clinicians
- Include dosing considerations
- Note any contraindications

### Future Directions
- Identify gaps in clinical knowledge
- Suggest areas for clinical trials
""",
                    "Research Focus": FINAL_ANALYSIS_PROMPT_TEMPLATE  # Default template
                })

        # --- Event Handlers for UI ---
        def initialize_credentials_and_proceed():
            """Initialize credentials from gcloud and proceed to setup."""
            global USER_CREDENTIALS
            
            # Try to get credentials from gcloud
            USER_CREDENTIALS = get_user_credentials_from_gcloud()
            
            if USER_CREDENTIALS:
                return gr.update(selected=1)
            else:
                return gr.update()  # Stay on current tab if auth fails
        
        # Start button handler
        start_button.click(initialize_credentials_and_proceed, outputs=[tabs])
        
        def show_create_project_form():
            accounts = list_billing_accounts()
            return gr.update(visible=True), gr.update(choices=accounts, value=accounts[0] if accounts else None), gr.update(visible=False)

        def hide_create_project_form():
            return gr.update(visible=False), gr.update(visible=True)

        def handle_billing_selection(billing_account):
            if billing_account is None:
                # Don't change anything when None is selected
                return gr.update(), gr.update()
            if billing_account == CREATE_BILLING_ACCOUNT_OPTION:
                # Clear the dropdown selection and return a status message
                status_msg = f"\n\n📋 **To create a billing account:**\n\n1. Open this link in your browser: {CREATE_BILLING_ACCOUNT_URL}\n2. Complete the billing account setup\n3. Restart the Gradio app and select your new billing account from the dropdown\n\n"
                return gr.update(value=None), gr.update(value=status_msg, visible=True)
            # Valid billing account selected, hide the message
            return gr.update(), gr.update(visible=False)

        def handle_project_creation(project_id, billing_account, model_dropdown, thinking_budget_slider, progress=gr.Progress()):
            # Get the model endpoint from the dropdown selection
            model_endpoint = MODEL_OPTIONS[model_dropdown]
            status, new_project_selection = create_new_project(project_id, billing_account, model_endpoint, thinking_budget_slider, progress)
            if new_project_selection:
                projects = list_projects()
                choices = [f"{p['name']} ({p['id']})" for p in projects]
                return gr.update(visible=False), gr.update(visible=True), gr.update(choices=choices, value=new_project_selection), status, gr.update(selected=2)
            return gr.update(), gr.update(), gr.update(), status, gr.update()

        # Tab Switching
        start_button.click(lambda: gr.update(selected=1), None, tabs)

        # Setup Tab Interactions
        create_project_btn.click(show_create_project_form, outputs=[create_project_box, billing_account_dropdown, setup_details_box])
        cancel_create_project_btn.click(hide_create_project_form, outputs=[create_project_box, setup_details_box])
        billing_account_dropdown.change(handle_billing_selection, inputs=[billing_account_dropdown], outputs=[billing_account_dropdown, billing_link_message])
        create_project_submit_btn.click(
            handle_project_creation, 
            inputs=[new_project_id_input, billing_account_dropdown, model_dropdown, thinking_budget_slider], 
            outputs=[create_project_box, setup_details_box, project_dropdown, status_output, tabs]
        )
        def handle_billing_setup_selection(billing_account):
            """Handle billing account selection in the billing setup box."""
            if billing_account is None:
                # Don't change anything when None is selected
                return gr.update(), gr.update()
            if billing_account == CREATE_BILLING_ACCOUNT_OPTION:
                # Clear the dropdown selection and return a status message
                status_msg = f"\n\n📋 **To create a billing account:**\n\n1. Open this link in your browser: {CREATE_BILLING_ACCOUNT_URL}\n2. Complete the billing account setup\n3. Restart the Gradio app and select your new billing account from the dropdown\n\n"
                return gr.update(value=None), gr.update(value=status_msg, visible=True)
            # Valid billing account selected, hide the message
            return gr.update(), gr.update(visible=False)

        def handle_link_billing(billing_account, project_dropdown, model_dropdown, thinking_budget_slider, progress=gr.Progress()):
            """Handle linking billing account to the project."""
            if not billing_account or billing_account == CREATE_BILLING_ACCOUNT_OPTION:
                return "❌ Please select a valid billing account.", gr.update(visible=True), gr.update(visible=False)
            
            project_id = project_dropdown.split('(')[-1].rstrip(')')
            progress(0.1, desc="Linking billing account...")
            
            success, message = link_billing_to_project(project_id, billing_account)
            if success:
                progress(0.3, desc="Billing linked! Continuing setup...")
                # After successful billing link, continue with the normal setup with model settings
                status, analyze_btn_update, tabs_update = proceed_with_project(project_dropdown, model_dropdown, thinking_budget_slider, progress)
                # Return appropriate updates for this function's outputs
                # The .then() chains will handle the analyze button and tabs updates based on the status message
                return status, gr.update(visible=False), gr.update(visible=True)
            else:
                return message, gr.update(visible=True), gr.update(visible=False)

        # State to track if we need billing setup
        needs_billing_setup = gr.State(False)
        
        # Modified proceed button click handler
        def handle_proceed_click(project_dropdown, model_dropdown, thinking_budget_slider, progress=gr.Progress()):
            """Handle the proceed button click."""
            status, analyze_btn_update, tabs_update = proceed_with_project(project_dropdown, model_dropdown, thinking_budget_slider, progress)
            
            if status == "billing_needed":
                # Show billing setup box and populate dropdown
                accounts = list_billing_accounts()
                return (
                    "❌ Billing is not enabled for this project. Please set up billing to continue.",
                    gr.update(interactive=False),  # analyze_btn
                    gr.update(),  # tabs (no change)
                    gr.update(visible=True),  # billing_setup_box
                    gr.update(visible=False),  # setup_details_box
                    gr.update(choices=accounts, value=accounts[0] if accounts else None),  # billing_setup_dropdown
                    True  # needs_billing_setup state
                )
            else:
                # Normal flow
                return (
                    status,
                    analyze_btn_update,
                    tabs_update,
                    gr.update(visible=False),  # billing_setup_box
                    gr.update(visible=True),  # setup_details_box
                    gr.update(),  # billing_setup_dropdown (no change)
                    False  # needs_billing_setup state
                )
        
        proceed_btn.click(
            handle_proceed_click, 
            inputs=[project_dropdown, model_dropdown, thinking_budget_slider], 
            outputs=[status_output, analyze_btn, tabs, billing_setup_box, setup_details_box, billing_setup_dropdown, needs_billing_setup]
        )

        # Billing setup handlers
        billing_setup_dropdown.change(
            handle_billing_setup_selection, 
            inputs=[billing_setup_dropdown], 
            outputs=[billing_setup_dropdown, billing_setup_message]
        )
        
        # Helper functions for the .then() chains
        def update_analyze_btn_based_on_status(status_markdown):
            """Update analyze button based on the status message."""
            # Extract the actual text value from the Markdown component data
            if isinstance(status_markdown, dict) and 'value' in status_markdown:
                status_text = status_markdown['value']
            elif isinstance(status_markdown, str):
                status_text = status_markdown
            else:
                status_text = str(status_markdown)
            
            return gr.update(interactive=status_text.startswith("✅"))
        
        def update_tabs_based_on_status(status_markdown):
            """Update tabs based on the status message."""
            # Extract the actual text value from the Markdown component data
            if isinstance(status_markdown, dict) and 'value' in status_markdown:
                status_text = status_markdown['value']
            elif isinstance(status_markdown, str):
                status_text = status_markdown
            else:
                status_text = str(status_markdown)
            
            if status_text.startswith("✅"):
                return gr.update(selected=2)
            else:
                return gr.update()
        
        link_billing_output = link_billing_btn.click(
            handle_link_billing,
            inputs=[billing_setup_dropdown, project_dropdown, model_dropdown, thinking_budget_slider],
            outputs=[status_output, billing_setup_box, setup_details_box]
        )
        
        # Update analyze button based on the status
        link_billing_output.then(
            update_analyze_btn_based_on_status,
            inputs=[status_output],
            outputs=[analyze_btn]
        )
        
        # Update tabs based on the status
        link_billing_output.then(
            update_tabs_based_on_status,
            inputs=[status_output],
            outputs=[tabs]
        )

        # Case Tab Interactions
        def show_extraction_loading():
            """Show loading state when extraction starts."""
            return gr.update(visible=True), gr.update(interactive=False)
        
        def extract_and_display(case_text, disease_prompt, events_prompt):
            """Extract medical information and display results using custom prompts."""
            if not case_text.strip():
                return (
                    gr.update(visible=False),  # extraction_box
                    "",  # disease_edit
                    "",  # events_edit
                    "",  # ai_extraction_display
                    {"extracted": False, "disease": "", "events": "", "ai_disease": "", "ai_events": ""},  # extraction_state
                    gr.update(interactive=False),  # proceed_button
                    "❌ Please enter case notes first.",  # case_status
                    gr.update(visible=False),  # extraction_loading
                    gr.update(interactive=True)  # extract_btn
                )
            
            if not genai_client:
                return (
                    gr.update(visible=True),
                    "",  # disease_edit
                    "",  # events_edit
                    "❌ Please complete setup first.",  # ai_extraction_display
                    {"extracted": False, "disease": "", "events": "", "ai_disease": "", "ai_events": ""},
                    gr.update(interactive=False),
                    "",
                    gr.update(visible=False),  # extraction_loading
                    gr.update(interactive=True)  # extract_btn
                )
            
            try:
                # Extract medical info with custom prompts
                medical_info = extract_medical_info(case_text, genai_client, disease_prompt, events_prompt)
                disease = medical_info.get('disease', '')
                events = medical_info.get('events', '')
                
                # Format display for AI extraction reference
                ai_display_text = f"""<div class="extraction-result">
<strong>AI Extracted Disease:</strong> {disease}

<strong>AI Extracted Events:</strong>
{events}
</div>"""
                
                return (
                    gr.update(visible=True),  # extraction_box
                    disease,  # disease_edit - populate with AI extraction
                    events,  # events_edit - populate with AI extraction
                    ai_display_text,  # ai_extraction_display
                    {"extracted": True, "disease": disease, "events": events, "ai_disease": disease, "ai_events": events},  # extraction_state
                    gr.update(interactive=True),  # proceed_button
                    "✅ Information extracted successfully. You can edit the values before proceeding.",  # case_status
                    gr.update(visible=False),  # extraction_loading
                    gr.update(interactive=True)  # extract_btn
                )
                
            except Exception as e:
                return (
                    gr.update(visible=True),
                    "",  # disease_edit
                    "",  # events_edit
                    f"❌ Error extracting information: {str(e)}",  # ai_extraction_display
                    {"extracted": False, "disease": "", "events": "", "ai_disease": "", "ai_events": ""},
                    gr.update(interactive=False),
                    "",
                    gr.update(visible=False),  # extraction_loading
                    gr.update(interactive=True)  # extract_btn
                )
        
        # Extract button click handler
        extract_btn.click(
            show_extraction_loading,
            outputs=[extraction_loading, extract_btn]
        ).then(
            extract_and_display,
            inputs=[case_input, disease_prompt_input, events_prompt_input],
            outputs=[extraction_box, disease_edit, events_edit, ai_extraction_display, extraction_state, proceed_to_persona_btn, case_status, extraction_loading, extract_btn]
        )
        
        # Enable/disable extract button based on case input
        def check_case_input(case_text):
            if case_text.strip():
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)
        
        case_input.change(
            check_case_input,
            inputs=[case_input],
            outputs=[extract_btn]
        )
        
        # Load example button handler
        def load_example_case():
            """Load example case notes and prompts."""
            return (
                SAMPLE_CASE,
                DISEASE_EXTRACTION_PROMPT,
                EVENT_EXTRACTION_PROMPT
            )

        load_example_btn.click(
            load_example_case,
            outputs=[case_input, disease_prompt_input, events_prompt_input]
        )
        
        # Event handlers for editable fields
        def update_disease(new_disease, extraction_state):
            """Update the disease value in extraction state when user edits it."""
            if extraction_state:
                extraction_state["disease"] = new_disease
            return extraction_state
        
        def update_events(new_events, extraction_state):
            """Update the events value in extraction state when user edits it."""
            if extraction_state:
                extraction_state["events"] = new_events
            return extraction_state
        
        def reset_disease(extraction_state):
            """Reset disease to the original AI extraction."""
            if extraction_state and "ai_disease" in extraction_state:
                return extraction_state["ai_disease"], extraction_state
            return "", extraction_state
        
        def reset_events(extraction_state):
            """Reset events to the original AI extraction."""
            if extraction_state and "ai_events" in extraction_state:
                return extraction_state["ai_events"], extraction_state
            return "", extraction_state
        
        # Connect the handlers
        disease_edit.change(
            update_disease,
            inputs=[disease_edit, extraction_state],
            outputs=[extraction_state]
        )
        
        events_edit.change(
            update_events,
            inputs=[events_edit, extraction_state],
            outputs=[extraction_state]
        )
        
        reset_disease_btn.click(
            reset_disease,
            inputs=[extraction_state],
            outputs=[disease_edit, extraction_state]
        )
        
        reset_events_btn.click(
            reset_events,
            inputs=[extraction_state],
            outputs=[events_edit, extraction_state]
        )

        # Modified proceed button handler to go to Persona tab
        def proceed_to_persona(case_text, extraction_state):
            if not case_text.strip():
                return "❌ Please enter case notes first.", gr.update(interactive=False), gr.update()
            if not extraction_state.get("extracted", False):
                return "❌ Please wait for information extraction to complete.", gr.update(interactive=False), gr.update()
            return "✅ Case notes and extracted information saved. Please customize your persona.", gr.update(interactive=True), gr.update(selected=3)

        proceed_to_persona_btn.click(
            proceed_to_persona,
            inputs=[case_input, extraction_state],
            outputs=[case_status, analyze_btn, tabs]
        )

        # Persona Tab Interactions
        def load_example_persona(personas):
            """Load a random example persona."""
            import random
            persona_name = random.choice(list(personas.keys()))
            return personas[persona_name]
        
        load_persona_btn.click(
            load_example_persona,
            inputs=[example_personas],
            outputs=[persona_text]
        )
        
        # Default criteria configuration
        DEFAULT_CRITERIA = [
            {"name": "disease_match", "description": "Does the article match the patient's disease?", "weight": 80, "type": "boolean", "deletable": True},
            {"name": "treatment_shown", "description": "Does the article show positive treatment results?", "weight": 50, "type": "boolean", "deletable": True},
            {"name": "pediatric_focus", "description": "Does the article focus on pediatric patients?", "weight": 60, "type": "boolean", "deletable": True},
            {"name": "clinical_trial", "description": "Is this a clinical trial?", "weight": 70, "type": "boolean", "deletable": True},
            {"name": "novelty", "description": "Does the article present novel findings?", "weight": 65, "type": "boolean", "deletable": True},
            {"name": "actionable_events_match", "description": "How many actionable events from the patient's case are mentioned in this article?", "weight": 100, "type": "numeric", "deletable": True},
            {"name": "human_clinical_data", "description": "Does the article include human clinical data?", "weight": 30, "type": "boolean", "deletable": True},
            {"name": "cell_studies", "description": "Does the article include cell studies?", "weight": 5, "type": "boolean", "deletable": True},
            {"name": "mice_studies", "description": "Does the article include mice studies?", "weight": 10, "type": "boolean", "deletable": True},
            {"name": "journal_impact", "description": "Journal impact factor (SJR)", "weight": 10, "type": "special_journal", "deletable": True},
            {"name": "year", "description": "Publication year penalty", "weight": 30, "type": "special_year", "deletable": True}
        ]
        
        def calculate_total_weight(criteria_list):
            """Calculate total weight from criteria list."""
            total = sum(c['weight'] for c in criteria_list if c is not None)
            return f"**Total Weight:** {total}"
        
        def update_criteria_display(criteria_list):
            """Update all criterion slot displays based on the criteria list."""
            updates = []
            
            # Type badge styling
            type_colors = {
                'boolean': '🟢',
                'numeric': '🔵', 
                'direct': '🟠',
                'special_journal': '🟣',
                'special_year': '🔴'
            }
            
            for i in range(20):
                if i < len(criteria_list) and criteria_list[i] is not None:
                    criterion = criteria_list[i]
                    # Row visibility
                    updates.append(gr.update(visible=True))
                    # Description
                    updates.append(gr.update(value=f"**{criterion['description']}**"))
                    # Type info
                    type_emoji = type_colors.get(criterion['type'], '⚫')
                    updates.append(gr.update(value=f"{type_emoji} Type: `{criterion['type']}` | Name: `{criterion['name']}`"))
                    # Slider value
                    updates.append(gr.update(value=criterion['weight']))
                    # Delete button
                    if criterion.get('deletable', True):
                        updates.append(gr.update(visible=True, interactive=True))
                    else:
                        updates.append(gr.update(visible=False))
                else:
                    # Hide this slot
                    updates.append(gr.update(visible=False))
                    updates.append(gr.update(value=""))
                    updates.append(gr.update(value=""))
                    updates.append(gr.update(value=0))
                    updates.append(gr.update(visible=False))
            
            # Add total weight
            updates.append(calculate_total_weight(criteria_list))
            
            return updates
        
        def show_add_criterion_dialog(criteria_list):
            """Show the dialog for adding a new criterion if there's space."""
            if len([c for c in criteria_list if c is not None]) >= 20:
                return gr.update(visible=False), gr.update(visible=True, value="⚠️ Maximum 20 criteria reached")
            return gr.update(visible=True), gr.update(visible=False)
        
        def hide_add_criterion_dialog():
            """Hide the add criterion dialog."""
            return gr.update(visible=False), "", "boolean", 10
        
        def add_new_criterion_with_details(criteria_list, description, criterion_type, weight):
            """Add a new custom criterion with user-provided details."""
            if not description.strip():
                return [criteria_list] + update_criteria_display(criteria_list) + [gr.update(visible=True)]
            
            # Check if we have space
            if len([c for c in criteria_list if c is not None]) >= 20:
                return [criteria_list] + update_criteria_display(criteria_list) + [gr.update(visible=True)]
            
            # Generate a safe name from description
            import re
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', description.lower())[:30]
            if not safe_name or safe_name[0].isdigit():
                safe_name = f"custom_{len([c for c in criteria_list if c is not None])}"
            
            new_criterion = {
                "name": safe_name,
                "description": description.strip(),
                "weight": weight,
                "type": criterion_type,
                "deletable": True,
                "user_defined": True
            }
            
            # Add to the list
            new_list = criteria_list.copy()
            new_list.append(new_criterion)
            
            return [new_list] + update_criteria_display(new_list) + [gr.update(visible=False)]
        
        def update_criterion_weight(criteria_list, slot_index, new_weight):
            """Update the weight for a specific criterion."""
            if slot_index < len(criteria_list) and criteria_list[slot_index] is not None:
                new_list = criteria_list.copy()
                new_list[slot_index]['weight'] = new_weight
                return new_list, calculate_total_weight(new_list)
            return criteria_list, calculate_total_weight(criteria_list)
        
        def delete_criterion(criteria_list, slot_index):
            """Delete a criterion from a specific slot."""
            if slot_index < len(criteria_list) and criteria_list[slot_index] is not None:
                if criteria_list[slot_index].get('deletable', True):
                    new_list = criteria_list.copy()
                    new_list.pop(slot_index)
                    return [new_list] + update_criteria_display(new_list)
            return [criteria_list] + update_criteria_display(criteria_list)
        
        def reset_to_defaults():
            """Reset criteria to defaults."""
            return [DEFAULT_CRITERIA.copy()] + update_criteria_display(DEFAULT_CRITERIA.copy())
        
        # Initialize criteria state on load
        def initialize_criteria():
            criteria = DEFAULT_CRITERIA.copy()
            return [criteria] + update_criteria_display(criteria)
        
        # Set up initial criteria
        all_outputs = [criteria_state]
        for i in range(20):
            all_outputs.extend([
                criterion_rows[i],
                criterion_descriptions[i],
                criterion_type_infos[i],
                criterion_sliders[i],
                criterion_delete_btns[i]
            ])
        all_outputs.append(total_weight_display)
        
        demo.load(
            initialize_criteria,
            outputs=all_outputs
        )
        
        # Criteria management event handlers
        add_criterion_btn.click(
            show_add_criterion_dialog,
            inputs=[criteria_state],
            outputs=[add_criterion_dialog, total_weight_display]
        )
        
        cancel_add_btn.click(
            hide_add_criterion_dialog,
            outputs=[add_criterion_dialog, new_criterion_description, new_criterion_type, new_criterion_weight]
        )
        
        confirm_add_btn.click(
            add_new_criterion_with_details,
            inputs=[criteria_state, new_criterion_description, new_criterion_type, new_criterion_weight],
            outputs=[criteria_state] + all_outputs[1:] + [add_criterion_dialog]
        ).then(
            lambda: ("", "boolean", 10),
            outputs=[new_criterion_description, new_criterion_type, new_criterion_weight]
        )
        
        reset_criteria_btn.click(
            reset_to_defaults,
            outputs=all_outputs
        )
        
        # Set up weight slider handlers for each slot
        for i in range(20):
            criterion_sliders[i].change(
                lambda weight, state, idx=i: update_criterion_weight(state, idx, weight),
                inputs=[criterion_sliders[i], criteria_state],
                outputs=[criteria_state, total_weight_display]
            )
        
        # Set up delete button handlers for each slot
        for i in range(20):
            criterion_delete_btns[i].click(
                lambda state, idx=i: delete_criterion(state, idx),
                inputs=[criteria_state],
                outputs=all_outputs
            )
        
        # Generator function for two-phase progressive analysis
        def run_two_phase_analysis(case_text, analysis_config, persona, criteria, extraction_state):
            """Generator that yields analysis progress updates with two-phase approach."""
            if not genai_client or not bq_client:
                yield {"status": "error", "message": "❌ Please complete setup first."}
                return
            
            try:
                # Get configuration
                default_articles = int(analysis_config.get("default_articles", 5))
                min_per_event = int(analysis_config.get("min_per_event", 3))
                max_articles = int(analysis_config.get("max_articles", 50))
                
                # Extract disease and events with IDs
                disease = extraction_state.get('disease', '')
                events_list = extraction_state.get('events_list', [])
                events_with_ids = extraction_state.get('events_with_ids', {})
                
                if not events_with_ids:
                    # Fallback: generate IDs if not present
                    events_text = extraction_state.get('events', '')
                    if '"' in events_text:
                        import re
                        events_list = re.findall(r'"([^"]+)"', events_text)
                    else:
                        events_list = [e.strip() for e in events_text.split('\n') if e.strip()]
                    
                    events_with_ids = {f"event_{i}": event for i, event in enumerate(events_list, 1)}
                    extraction_state['events_list'] = events_list
                    extraction_state['events_with_ids'] = events_with_ids
                
                yield {"status": "starting", "message": "Starting two-phase analysis..."}
                
                # Phase 1: Event Coverage Check
                yield {"status": "phase1_start", "message": "Phase 1: Checking event coverage..."}
                
                event_coverage = {event_id: [] for event_id in events_with_ids.keys()}
                total_articles_searched = 0
                selected_pmids = set()
                all_searched_articles = []  # Keep all articles for potential full analysis
                
                embedding_model_path = f"{PROJECT_ID}.{USER_DATASET}.textembed"
                
                while total_articles_searched < max_articles:
                    # Check if all events have minimum coverage
                    all_covered = all(len(pmids) >= min_per_event for pmids in event_coverage.values())
                    if all_covered:
                        break
                    
                    # Search next batch
                    batch_start = total_articles_searched + 1
                    batch_end = min(total_articles_searched + default_articles, max_articles)
                    
                    yield {
                        "status": "phase1_searching",
                        "message": f"Searching articles {batch_start}-{batch_end}...",
                        "coverage": event_coverage
                    }
                    
                    # Fetch articles with offset
                    articles_df = search_pubmed_articles(
                        disease, events_list, bq_client, embedding_model_path, 
                        PUBMED_TABLE, default_articles, offset=total_articles_searched
                    )
                    
                    if articles_df.empty:
                        break
                    
                    all_searched_articles.append(articles_df)
                    
                    # Analyze batch for event coverage
                    yield {
                        "status": "phase1_analyzing",
                        "message": f"Analyzing batch for event coverage...",
                        "coverage": event_coverage
                    }
                    
                    coverage_results = analyze_event_coverage_batch(
                        articles_df, disease, events_with_ids, bq_client
                    )
                    
                    # Update coverage tracking using PMCID
                    for result in coverage_results:
                        pmcid = result['PMCID']
                        event_ids_str = result.get('event_ids', '')
                        
                        if event_ids_str:
                            event_ids = [e.strip() for e in event_ids_str.split(',') if e.strip()]
                            for event_id in event_ids:
                                if event_id in event_coverage:
                                    if pmcid not in event_coverage[event_id]:
                                        event_coverage[event_id].append(pmcid)
                                    selected_pmids.add(pmcid)
                    
                    total_articles_searched += len(articles_df)
                    
                    # Report coverage status
                    coverage_status = []
                    for event_id, event_text in events_with_ids.items():
                        count = len(event_coverage[event_id])
                        status_text = f"{event_text[:30]}...: {count}/{min_per_event}"
                        if count >= min_per_event:
                            status_text = f"✓ {status_text}"
                        coverage_status.append(status_text)
                    
                    yield {
                        "status": "phase1_progress",
                        "message": f"Searched {total_articles_searched} articles",
                        "coverage": event_coverage,
                        "coverage_status": coverage_status
                    }
                
                # Phase 1 Complete
                all_covered = all(len(pmids) >= min_per_event for pmids in event_coverage.values())
                phase1_summary = f"Phase 1 complete: Searched {total_articles_searched} articles, found {len(selected_pmids)} relevant articles"
                
                if not all_covered:
                    phase1_summary += " (Warning: Some events did not meet minimum coverage)"
                
                yield {
                    "status": "phase1_complete",
                    "message": phase1_summary,
                    "selected_articles": len(selected_pmids),
                    "coverage": event_coverage
                }
                
                # Phase 2: Full Analysis - Analyze ALL articles
                # Combine all searched articles
                all_articles_df = pd.concat(all_searched_articles, ignore_index=True) if all_searched_articles else pd.DataFrame()
                
                if all_articles_df.empty:
                    yield {
                        "status": "complete",
                        "message": "No articles found",
                        "results": {
                            'articles': [],
                            'disease': disease,
                            'events': events_list,
                            'case_text': case_text,
                            'persona': persona,
                            'criteria': criteria
                        }
                    }
                    return
                
                yield {"status": "phase2_start", "message": f"Phase 2: Performing full analysis on ALL {len(all_articles_df)} articles found..."}
                
                # Use all articles, not just selected ones
                selected_articles_df = all_articles_df
                
                # Analyze selected articles one by one for UI updates
                analyzed_articles = []
                
                for idx, (_, article_row) in enumerate(selected_articles_df.iterrows()):
                    yield {
                        "status": "phase2_analyzing",
                        "current": idx + 1,
                        "total": len(selected_articles_df),
                        "message": f"Full analysis: article {idx + 1} of {len(selected_articles_df)}..."
                    }
                    
                    try:
                        # Create single-row DataFrame
                        single_article_df = pd.DataFrame([article_row])
                        
                        # Full analysis with all criteria
                        analysis_results = analyze_article_batch_with_criteria(
                            single_article_df, disease, events_list, bq_client, 
                            journal_impact_dict, persona, criteria
                        )
                        
                        if analysis_results and len(analysis_results) > 0:
                            analysis = analysis_results[0]
                            
                            # Merge analysis results with article data
                            article_data = article_row.to_dict()
                            article_data.update(analysis)
                            
                            # Calculate score and get breakdown
                            score, point_breakdown = calculate_dynamic_score(analysis, criteria, journal_impact_dict)
                            
                            article_data['score'] = score
                            article_data['point_breakdown'] = point_breakdown
                            article_data['content'] = article_row.get('content', '')
                            article_data['pmcid'] = article_row.get('PMCID', '')
                            article_data['pmid'] = article_row.get('PMID', '')  # May be None
                            
                            analyzed_articles.append(article_data)
                            
                            # Yield the analyzed article
                            yield {
                                "status": "article_complete",
                                "current": idx + 1,
                                "total": len(selected_articles_df),
                                "article": {
                                    "score": score,
                                    "point_breakdown": point_breakdown,
                                    "title": analysis.get('title', 'Unknown Title'),
                                    "journal": analysis.get('journal_title', 'Unknown Journal'),
                                    "year": analysis.get('year', 'N/A'),
                                    "pmid": article_row.get('PMID', ''),
                                    "pmcid": article_row.get('PMCID', ''),
                                    "content": article_row.get('content', ''),
                                    "metadata": analysis
                                }
                            }
                        else:
                            yield {
                                "status": "article_failed",
                                "current": idx + 1,
                                "total": len(selected_articles_df),
                                "message": f"Failed to analyze article {idx + 1}"
                            }
                            
                    except Exception as e:
                        print(f"Error analyzing article {idx + 1}: {str(e)}")
                        yield {
                            "status": "article_failed",
                            "current": idx + 1,
                            "total": len(selected_articles_df),
                            "message": f"Error analyzing article {idx + 1}: {str(e)}"
                        }
                    
                    # Add a small delay to avoid rate limiting
                    if idx < len(selected_articles_df) - 1:
                        time.sleep(1)
                
                # Final results
                yield {
                    "status": "complete",
                    "message": f"✅ Analysis complete! Analyzed {len(analyzed_articles)} articles.",
                    "results": {
                        'articles': analyzed_articles,
                        'disease': disease,
                        'events': events_list,
                        'case_text': case_text,
                        'persona': persona,
                        'criteria': criteria,
                        'total_searched': total_articles_searched,
                        'event_coverage': event_coverage
                    }
                }
                
            except Exception as e:
                yield {
                    "status": "error",
                    "message": f"❌ Error during analysis: {str(e)}"
                }
        
        # Helper function to generate article HTML card
        def generate_article_html(article, idx):
            """Generate consistent HTML for an article card."""
            # Format points breakdown with better visibility
            breakdown_items = []
            
            if article.get('point_breakdown'):
                for key, value in article.get('point_breakdown', {}).items():
                    formatted_key = key.replace('_', ' ').title()
                    if value > 0:
                        breakdown_items.append(f'<span style="color: #28a745; font-weight: bold;">{formatted_key}: +{value:.1f}</span>')
                    elif value < 0:
                        breakdown_items.append(f'<span style="color: #dc3545; font-weight: bold;">{formatted_key}: {value:.1f}</span>')
            
            # Get metadata
            metadata = article.get('metadata', article)
            
            # Create events display - handle string format
            events_html = ""
            actionable_events = metadata.get('actionable_events', '')
            if actionable_events:
                if isinstance(actionable_events, str):
                    # Parse JSON string or split by comma
                    try:
                        import json as json_module
                        events_list = json_module.loads(actionable_events)
                    except:
                        # Treat as comma-separated
                        events_list = [e.strip() for e in actionable_events.split(',') if e.strip()]
                else:
                    events_list = actionable_events
                
                # Process events
                if isinstance(events_list, list):
                    for event in events_list:
                        if isinstance(event, dict):
                            event_text = event.get('event', '')
                            matches = event.get('matches_query', False)
                            color = '#28a745' if matches else '#6c757d'
                            weight = 'bold' if matches else 'normal'
                            events_html += f'<span style="color: {color}; font-weight: {weight}; margin-right: 10px;">{event_text}</span>'
                        else:
                            # Simple string
                            events_html += f'<span style="color: #6c757d; margin-right: 10px;">{event}</span>'
            
            # Paper type and other metadata
            paper_type = metadata.get('paper_type', 'Unknown')
            
            return f"""
            <div class='article-card'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <h4 style='margin-top: 0; flex: 1; color: #212529;'>{idx + 1}. {article.get('title', 'Unknown')}</h4>
                    <div style='text-align: right;'>
                        <span style='font-size: 32px; font-weight: bold; color: #007bff; display: block;'>{article.get('score', 0):.1f}</span>
                        <span style='font-size: 14px; color: #6c757d;'>Total Score</span>
                    </div>
                </div>
                
                <div class='score-breakdown-box'>
                    <h5 style='margin-top: 0; color: #495057;'>📊 Score Breakdown</h5>
                    <div style='margin: 10px 0;'>
                        {' | '.join(breakdown_items) if breakdown_items else '<span style="color: #6c757d;">No scoring criteria matched</span>'}
                    </div>
                </div>
                
                <div style='margin: 10px 0; color: #212529;'>
                    <strong style='color: #212529;'>Journal:</strong> {article.get('journal_title', metadata.get('journal_title', 'Unknown'))} | 
                    <strong style='color: #212529;'>Year:</strong> {article.get('year', metadata.get('year', 'N/A'))} | 
                    <strong style='color: #212529;'>Type:</strong> {paper_type}
                </div>
                
                <div style='margin: 10px 0;'>
                    <strong style='color: #212529;'>Links:</strong> 
                    <a href="https://pmc.ncbi.nlm.nih.gov/articles/{article.get('pmcid', '')}/" target="_blank" style="color: #0066cc; text-decoration: underline; margin-right: 15px;">
                        🔗 PMC Full Text (PMCID: {article.get('pmcid', 'N/A')})
                    </a>
                    {f'<a href="https://pubmed.ncbi.nlm.nih.gov/{article.get("pmid")}/" target="_blank" style="color: #0066cc; text-decoration: underline;">📄 PubMed (PMID: {article.get("pmid")})</a>' if article.get('pmid') else ''}
                </div>
                
                <div style='margin: 10px 0;'>
                    <strong style='color: #212529;'>Actionable Events Found:</strong><br/>
                    <div style='margin-top: 5px;'>
                        {events_html if events_html else '<span style="color: #6c757d;">None found</span>'}
                    </div>
                </div>
                
                <details style='margin-top: 15px;'>
                    <summary style="cursor: pointer; color: #0066cc; font-weight: bold;">📑 View Full Article</summary>
                    <div class='article-content-box'>
                        <pre>{article.get('content', 'No content available')}</pre>
                    </div>
                </details>
            </div>
            """
        
        # Function to update the UI based on generator output
        def update_analysis_display(progress_data, current_results, is_active):
            """Update the display based on progress data."""
            if not progress_data:
                return [current_results, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), is_active]
            
            status = progress_data.get("status", "")
            
            if status == "error":
                return [
                    current_results,
                    gr.update(value=progress_data.get("message", "Error")),
                    gr.update(),
                    gr.update(),
                    gr.update(visible=True, value=progress_data.get("message", "")),
                    gr.update(visible=False),
                    False
                ]
            
            elif status == "starting":
                return [
                    [],  # Clear results
                    gr.update(value="🔄 Starting analysis..."),
                    gr.update(value=pd.DataFrame()),
                    gr.update(value=""),
                    gr.update(visible=False),
                    gr.update(visible=True),
                    True
                ]
            
            elif status == "searching":
                return [
                    current_results,
                    gr.update(value=f"🔍 {progress_data.get('message', '')}"),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    is_active
                ]
            
            elif status == "search_complete":
                return [
                    current_results,
                    gr.update(value=f"📚 {progress_data.get('message', '')}"),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    is_active
                ]
            
            elif status == "analyzing":
                current = progress_data.get("current", 0)
                total = progress_data.get("total", 0)
                progress_pct = (current / total * 100) if total > 0 else 0
                return [
                    current_results,
                    gr.update(value=f"🔬 Analyzing article {current}/{total} ({progress_pct:.0f}%)"),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    is_active
                ]
            
            elif status == "article_complete":
                # Add the new article to results
                new_results = current_results.copy() if current_results else []
                article = progress_data.get("article", {})
                new_results.append(article)
                
                # Sort by score descending
                new_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Create DataFrame for display
                df_data = []
                for r in new_results:
                    df_data.append({
                        "Score": f"{r.get('score', 0):.1f}",
                        "Title": r.get('title', 'Unknown')[:80] + "..." if len(r.get('title', '')) > 80 else r.get('title', 'Unknown'),
                        "Journal": r.get('journal', 'Unknown'),
                        "Year": r.get('year', 'N/A'),
                        "Status": "✅ Analyzed"
                    })
                
                display_df = pd.DataFrame(df_data)
                
                # Update detailed analysis HTML using consistent generator
                detailed_html = "<div style='max-height: 800px; overflow-y: auto;'>"
                for idx, r in enumerate(new_results[:10]):  # Show top 10 in detail
                    detailed_html += generate_article_html(r, idx)
                detailed_html += "</div>"
                
                current = progress_data.get("current", 0)
                total = progress_data.get("total", 0)
                
                return [
                    new_results,
                    gr.update(value=f"🔬 Analyzed {current}/{total} articles"),
                    gr.update(value=display_df),
                    gr.update(value=detailed_html),
                    gr.update(),
                    gr.update(),
                    is_active
                ]
            
            elif status == "article_failed":
                current = progress_data.get("current", 0)
                total = progress_data.get("total", 0)
                return [
                    current_results,
                    gr.update(value=f"⚠️ Article {current}/{total} failed - {progress_data.get('message', '')}"),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    is_active
                ]
            
            elif status == "complete":
                # Final update with all results
                results = progress_data.get("results", {})
                articles = results.get("articles", [])
                
                # Sort articles by score
                articles.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                # Create final DataFrame with interactive elements
                df_data = []
                for r in articles:
                    df_data.append({
                        "Score": f"{r.get('score', 0):.1f}",
                        "PMID": r.get("pmid", "N/A"),
                        "Title": r.get('title', 'Unknown')[:80] + "..." if len(r.get('title', '')) > 80 else r.get('title', 'Unknown'),
                        "Journal": r.get('journal_title', 'Unknown'),
                        "Year": r.get('year', 'N/A'),
                        "Status": "✅ Complete"
                    })
                
                display_df = pd.DataFrame(df_data)
                
                # Create enhanced detailed HTML using consistent generator
                detailed_html = "<div style='max-height: 800px; overflow-y: auto;'>"
                detailed_html += "<h3>Detailed Results</h3>"
                
                for idx, article in enumerate(articles):
                    detailed_html += generate_article_html(article, idx)
                
                detailed_html += "</div>"
                
                # Summary message
                summary = f"""
                ### Analysis Summary
                
                - **Disease:** {results.get('disease', 'Unknown')}
                - **Events:** {', '.join(results.get('events', []))}
                - **Articles Analyzed:** {len(articles)}
                """
                
                # Only add top score if articles exist
                if articles and len(articles) > 0:
                    summary += f"\n- **Top Score:** {articles[0].get('score', 0):.1f}"
                else:
                    summary += f"\n- **Top Score:** N/A"
                
                if articles:
                    summary += "\n\n#### Top 5 Articles:\n"
                    for i, article in enumerate(articles[:5]):
                        summary += f"\n{i+1}. **{article.get('title', 'Unknown')}** (Score: {article.get('score', 0):.1f})"
                        if article.get('pmid'):
                            summary += f" - [PubMed](https://pubmed.ncbi.nlm.nih.gov/{article.get('pmid')}/)"
                else:
                    summary += "\n\n⚠️ No articles were successfully analyzed. This may be due to API quota limits or processing errors."
                
                # Debug log
                print(f"DEBUG: Analysis complete with {len(articles)} articles")
                
                return [
                    articles,
                    gr.update(value=progress_data.get("message", "")),
                    gr.update(value=display_df),
                    gr.update(value=detailed_html),
                    gr.update(visible=True),  # Show final_analysis_btn
                    gr.update(visible=False),
                    False
                ]
            
            return [current_results, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), is_active]
        
        # New functions to handle configuration updates
        def update_analysis_config(default_articles, min_per_event):
            """Update the analysis configuration state."""
            return {
                "default_articles": int(default_articles),
                "min_per_event": int(min_per_event),
                "max_articles": 50
            }
        
        # Connect configuration inputs to state
        default_articles_input.change(
            update_analysis_config,
            inputs=[default_articles_input, min_articles_per_event],
            outputs=[analysis_config]
        )
        
        min_articles_per_event.change(
            update_analysis_config,
            inputs=[default_articles_input, min_articles_per_event],
            outputs=[analysis_config]
        )
        
        # Function to run the two-phase analysis with proper generator handling
        def run_full_analysis_two_phase(case_text, analysis_config, persona, criteria, extraction_state):
            """Run the two-phase analysis with progressive updates."""
            # Initialize states
            results = []
            is_active = True
            
            # Create the generator for two-phase analysis
            generator = run_two_phase_analysis(case_text, analysis_config, persona, criteria, extraction_state)
            
            # Process each yield from the generator
            for progress_data in generator:
                if not is_active:  # Check if analysis was stopped
                    break
                    
                # Update the display based on status
                status = progress_data.get("status", "")
                
                # Handle phase 1 specific statuses
                if status in ["phase1_start", "phase1_searching", "phase1_analyzing", "phase1_progress", "phase1_complete"]:
                    if status == "phase1_progress":
                        # Format coverage status for display
                        coverage_status = progress_data.get("coverage_status", [])
                        coverage_html = "<br>".join(coverage_status)
                        yield [
                            results,
                            gr.update(value=f"{progress_data.get('message', '')}"),
                            gr.update(visible=True, value=coverage_html),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(visible=True),
                            is_active
                        ]
                    else:
                        yield [
                            results,
                            gr.update(value=progress_data.get("message", "")),
                            gr.update(visible=True),
                            gr.update(),
                            gr.update(),
                            gr.update(),
                            gr.update(visible=True),
                            is_active
                        ]
                elif status == "phase2_start":
                    yield [
                        results,
                        gr.update(value=progress_data.get("message", "")),
                        gr.update(visible=False),  # Hide coverage display for phase 2
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(visible=True),
                        is_active
                    ]
                elif status == "phase2_analyzing":
                    current = progress_data.get("current", 0)
                    total = progress_data.get("total", 0)
                    progress_pct = (current / total * 100) if total > 0 else 0
                    yield [
                        results,
                        gr.update(value=f"Phase 2: Analyzing article {current}/{total} ({progress_pct:.0f}%)"),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        gr.update(),
                        is_active
                    ]
                else:
                    # Use the regular update display for other statuses
                    results, progress_update, df_update, html_update, summary_update, stop_btn_update, is_active = update_analysis_display(
                        progress_data, results, is_active
                    )
                    
                    # Add event coverage display update
                    yield [
                        results,
                        progress_update,
                        gr.update(),  # event_coverage_display
                        df_update,
                        html_update,
                        summary_update,
                        stop_btn_update,
                        is_active
                    ]
        
        # Original analyze button (from Persona tab) just switches to Results tab
        analyze_btn.click(
            lambda: gr.update(selected=4),  # Switch to Results tab
            outputs=[tabs]
        )
        
        # Handler for Start Analysis button in Results tab (two-phase analysis)
        def start_two_phase_analysis(case_text, analysis_config, persona, criteria, extraction_state):
            """Start the two-phase analysis."""
            if not extraction_state.get("extracted", False):
                return (
                    gr.update(value="❌ Please extract disease and events first."),
                    gr.update(),
                    gr.update(),
                    gr.update()
                )
            
            return (
                gr.update(value="🔄 Starting two-phase analysis..."),  # analysis_progress
                gr.update(visible=True),  # stop_analysis_btn
                gr.update(visible=True, value=""),  # event_coverage_display
                gr.update()  # other updates
            )
        
        start_analysis_btn.click(
            start_two_phase_analysis,
            inputs=[case_input, analysis_config, persona_text, criteria_state, extraction_state],
            outputs=[analysis_progress, stop_analysis_btn, event_coverage_display, live_results_df]
        ).then(
            run_full_analysis_two_phase,
            inputs=[case_input, analysis_config, persona_text, criteria_state, extraction_state],
            outputs=[results_state, analysis_progress, event_coverage_display, live_results_df, detailed_analysis_html, final_analysis_btn, stop_analysis_btn, analysis_active]
        )
        
        # Stop button handler
        def stop_analysis():
            """Stop the ongoing analysis."""
            return (
                False,  # Set analysis_active to False
                gr.update(value="⏹️ Analysis stopped by user."),
                gr.update(visible=False)
            )
        
        stop_analysis_btn.click(
            stop_analysis,
            outputs=[analysis_active, analysis_progress, stop_analysis_btn]
        )
        
        # Final Analysis tab handlers
        def prepare_final_analysis_tab(results):
            """Prepare the final analysis tab with article selection."""
            if not results:
                return (
                    gr.update(choices=[], value=[]),  # article_selections
                    gr.update(value="No articles available"),  # selected_count
                    gr.update(value="Estimated tokens: 0"),  # token_estimate
                    gr.update(interactive=False),  # generate_final_btn
                    [],  # selected_articles_state
                    gr.update(visible=False)  # final_analysis_btn
                )
            
            # Create choices for checkbox group
            choices = []
            for idx, article in enumerate(results):
                metadata = article.get('metadata', article)
                choice_label = f"[Score: {article.get('score', 0):.1f}] {metadata.get('title', 'Unknown')[:100]}... ({metadata.get('journal_title', 'Unknown')}, {metadata.get('year', 'N/A')})"
                choices.append((choice_label, idx))
            
            # Select top 10 by default
            default_selected = list(range(min(10, len(results))))
            
            # Calculate initial token estimate
            selected_articles = [results[i] for i in default_selected]
            total_tokens = sum(estimate_tokens(article.get('content', '')) for article in selected_articles)
            
            return (
                gr.update(choices=choices, value=default_selected),  # article_selections
                gr.update(value=f"{len(default_selected)} articles selected"),  # selected_count
                gr.update(value=f"Estimated tokens: {total_tokens:,}"),  # token_estimate
                gr.update(interactive=len(default_selected) > 0),  # generate_final_btn
                selected_articles,  # selected_articles_state
                gr.update(visible=True)  # final_analysis_btn
            )
        
        def update_article_selection(selected_indices, all_results):
            """Update selection count and token estimate."""
            if not all_results or selected_indices is None:
                return (
                    gr.update(value="0 articles selected"),
                    gr.update(value="Estimated tokens: 0"),
                    gr.update(interactive=False),
                    []
                )
            
            selected_articles = [all_results[i] for i in selected_indices]
            total_tokens = sum(estimate_tokens(article.get('content', '')) for article in selected_articles)
            
            return (
                gr.update(value=f"{len(selected_indices)} articles selected"),
                gr.update(value=f"Estimated tokens: {total_tokens:,}"),
                gr.update(interactive=len(selected_indices) > 0),
                selected_articles
            )
        
        def select_all_articles(all_results):
            """Select all articles."""
            if not all_results:
                return gr.update()
            return gr.update(value=list(range(len(all_results))))
        
        def deselect_all_articles():
            """Deselect all articles."""
            return gr.update(value=[])
        
        def generate_final_analysis(case_text, extraction_state, selected_articles):
            """Generate the final analysis with streaming."""
            if not selected_articles:
                yield gr.update(value="❌ Please select at least one article.")
                return
            
            # Extract disease and events
            disease = extraction_state.get('disease', '')
            events = extraction_state.get('events_list', [])
            
            # Stream the analysis
            yield gr.update(value="🔄 Generating comprehensive analysis...")
            
            accumulated_text = ""
            for chunk in generate_final_analysis_stream(case_text, disease, events, selected_articles, genai_client):
                accumulated_text = chunk
                yield gr.update(value=accumulated_text)
        
        # Connect final analysis button to switch tabs and prepare selection
        final_analysis_btn.click(
            lambda results: prepare_final_analysis_tab(results),
            inputs=[results_state],
            outputs=[article_selections, selected_count, token_estimate, generate_final_btn, selected_articles_state, final_analysis_btn]
        ).then(
            lambda: gr.update(selected=5),
            outputs=[tabs]
        )
        
        # Update selection handlers
        article_selections.change(
            update_article_selection,
            inputs=[article_selections, results_state],
            outputs=[selected_count, token_estimate, generate_final_btn, selected_articles_state]
        )
        
        select_all_btn.click(
            select_all_articles,
            inputs=[results_state],
            outputs=[article_selections]
        )
        
        deselect_all_btn.click(
            deselect_all_articles,
            outputs=[article_selections]
        )
        
        # Prompt editing handlers
        def reset_analysis_prompt():
            """Reset the analysis prompt to default."""
            return gr.update(value=FINAL_ANALYSIS_PROMPT_TEMPLATE)
        
        def load_example_analysis_prompt(example_prompts):
            """Load a random example prompt."""
            import random
            prompt_name = random.choice(list(example_prompts.keys()))
            return gr.update(value=example_prompts[prompt_name])
        
        reset_prompt_btn.click(
            reset_analysis_prompt,
            outputs=[analysis_prompt_input]
        )
        
        load_example_prompt_btn.click(
            load_example_analysis_prompt,
            inputs=[example_prompts],
            outputs=[analysis_prompt_input]
        )
        
        # Modified generate_final_analysis to use custom prompt
        def generate_final_analysis_with_custom_prompt(case_text, extraction_state, selected_articles, custom_prompt):
            """Generate the final analysis with custom prompt and streaming."""
            if not selected_articles:
                yield gr.update(value="❌ Please select at least one article.")
                return
            
            # Extract disease and events
            disease = extraction_state.get('disease', '')
            events = extraction_state.get('events_list', [])
            
            # Create custom prompt by filling in the template
            sorted_articles = sorted(selected_articles, key=lambda x: x.get('score', 0), reverse=True)
            
            # Format all selected articles
            articles_content_parts = []
            for idx, article in enumerate(sorted_articles, 1):
                articles_content_parts.append(format_article_for_analysis(article, idx))
            
            # Join all articles with separator
            articles_content = ("\n" + "="*80 + "\n").join(articles_content_parts)
            
            # Fill in the custom template
            try:
                filled_prompt = custom_prompt.format(
                    case_description=case_text,
                    primary_focus=disease,
                    key_concepts=', '.join(events),
                    articles_content=articles_content
                )
            except KeyError as e:
                yield gr.update(value=f"❌ Error in prompt template: Missing placeholder {{{e}}}. Please check your prompt template.")
                return
            
            # Stream the analysis
            yield gr.update(value="🔄 Generating comprehensive analysis...")
            
            # Use the existing streaming logic but with custom prompt
            if not genai_client:
                yield gr.update(value="❌ Gemini client not initialized. Please complete setup first.")
                return
            
            try:
                # Generate content with streaming
                config = GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=8192,
                    candidate_count=1,
                    thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
                )
                
                # Stream the response
                try:
                    response_stream = genai_client.models.generate_content_stream(
                        model=MODEL_ID,
                        contents=[filled_prompt],
                        config=config
                    )
                    
                    accumulated_text = ""
                    
                    for chunk in response_stream:
                        if hasattr(chunk, 'candidates') and chunk.candidates:
                            for candidate in chunk.candidates:
                                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                                    for part in candidate.content.parts:
                                        if hasattr(part, 'text'):
                                            accumulated_text += part.text
                                            # Yield the accumulated content so far
                                            yield gr.update(value=accumulated_text)
                    
                    # If no content was generated
                    if not accumulated_text:
                        yield gr.update(value="❌ No analysis was generated. Please try again.")
                        
                except Exception as e:
                    yield gr.update(value=f"❌ Error during analysis generation: {str(e)}")
                
            except Exception as e:
                yield gr.update(value=f"❌ Error generating final analysis: {str(e)}")
        
        # Generate final analysis with custom prompt
        generate_final_btn.click(
            generate_final_analysis_with_custom_prompt,
            inputs=[case_input, extraction_state, selected_articles_state, analysis_prompt_input],
            outputs=[final_analysis_display]
        )

        # Initial load
        demo.load(get_initial_projects, outputs=[project_dropdown, status_output])

    return demo

def main():
    """Main function to run the app."""
    parser = argparse.ArgumentParser(description='PubMed Medical Literature Analysis App')
    parser.add_argument('--share', action='store_true', help='Share the app publicly')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the app on')
    parser.add_argument('--server-name', type=str, default='0.0.0.0', help='Server name/IP to bind to')
    
    args = parser.parse_args()
    
    print("🏥 PubMed Medical Literature Analysis App")
    print("==========================================")
    print("\n📋 Setup Instructions:")
    print("1. Make sure you have authenticated with Google Cloud:")
    print("   $ gcloud auth application-default login")
    print("\n2. Ensure you have the necessary permissions for:")
    print("   - BigQuery (bigquery.datasets.create, bigquery.models.create)")
    print("   - Vertex AI (aiplatform.endpoints.predict)")
    print("   - Resource Manager (resourcemanager.projects.list)")
    print("\n3. Launch the app and follow the setup wizard")
    print("\n==========================================\n")
    
    # Create and launch the app
    demo = create_app(share=args.share)
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name=args.server_name,
        debug=False
    )

if __name__ == "__main__":
    main()
