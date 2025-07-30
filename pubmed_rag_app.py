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
import plotly.express as px
import plotly.graph_objects as go
from google import genai
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
MODEL_ID = "gemini-2.5-flash"
JOURNAL_IMPACT_CSV_URL = "https://raw.githubusercontent.com/WandLZhang/scimagojr_2024/main/scimagojr_2024.csv"
REQUIRED_APIS = ["aiplatform.googleapis.com", "bigquery.googleapis.com", "cloudresourcemanager.googleapis.com"]
CREATE_BILLING_ACCOUNT_URL = "https://console.cloud.google.com/billing/create?inv=1&invt=Ab4E_Q"
CREATE_BILLING_ACCOUNT_OPTION = "→ Create New Billing Account"
SAMPLE_CASE = """A 4-year-old male presents with a 3-week history of progressive fatigue, pallor, and easy bruising. \n
Physical examination reveals hepatosplenomegaly and scattered petechiae. \n
\n
Laboratory findings:\n
- WBC: 45,000/μL with 80% blasts\n
- Hemoglobin: 7.2 g/dL\n
- Platelets: 32,000/μL\n
\n
Flow cytometry: CD33+, CD13+, CD117+, CD34+, HLA-DR+, CD19-, CD3-\n
\n
Cytogenetics: 46,XY,t(9;11)(p21.3;q23.3)\n
Molecular: KMT2A-MLLT3 fusion detected, FLT3-ITD positive, NRAS G12D mutation\n
\n
Diagnosis: KMT2A-rearranged acute myeloid leukemia (AML)"""

# --- Global Variables ---
genai_client, bq_client = None, None
journal_impact_dict = {}
PROJECT_ID = ""
LOCATION = "global"
USER_DATASET = "pubmed"

# Scoring presets
SCORING_PRESETS = {
    "Clinical Focus": {
        "disease_match": 50,
        "treatment_efficacy": 50,
        "clinical_trial": 40
    }
}

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

def create_new_project(project_id, billing_account_name, progress=gr.Progress()):
    """Creates a new GCP project, links billing, and enables necessary APIs."""
    global USER_CREDENTIALS
    try:
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

        # Use the shared setup logic
        global genai_client, bq_client, journal_impact_dict
        genai_client, bq_client, journal_impact_dict = setup_project(project_id, LOCATION, USER_DATASET, progress)

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
def setup_project(project_id, location, dataset, progress=gr.Progress()):
    """Common setup logic for both new and existing projects."""
    try:
        # Set environment variable
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id
        
        progress(0.7, desc="Initializing clients...")
        genai_client, bq_client = init_clients(project_id, location)
        if not genai_client or not bq_client:
            raise ConnectionError("Failed to initialize Google Cloud clients.")

        # Setup BigQuery dataset and model
        setup_bigquery(project_id, dataset, bq_client, progress)

        progress(0.9, desc="Loading journal data...")
        journal_impact_dict = load_journal_data()
        
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

def load_journal_data():
    try:
        df = pd.read_csv(JOURNAL_IMPACT_CSV_URL, sep=';')
        return dict(zip(df['Title'], df['SJR']))
    except:
        return {}

def extract_medical_info(case_text, client):
    prompts = {
        "disease": "Extract the primary disease diagnosis. Return ONLY the name.",
        "events": "Extract all actionable medical events (mutations, biomarkers, etc.). Return a comma-separated list."
    }
    results = {}
    for key, prompt in prompts.items():
        full_prompt = f"{prompt}\n\nCase notes:\n{case_text}"
        response = client.models.generate_content(model=MODEL_ID, contents=[full_prompt], config=GenerateContentConfig(temperature=0))
        results[key] = response.text.strip()
    return results

def search_pubmed_articles(disease, events, bq_client, embedding_model, pubmed_table, top_k):
    query_text = f"{disease} {' '.join(events)}"
    # Debug: print the project being used
    print(f"Using BigQuery project: {bq_client.project}")
    print(f"Using embedding model: {embedding_model}")
    
    sql = f"""SELECT base.PMID, base.content, base.text as abstract, distance FROM VECTOR_SEARCH(TABLE `{pubmed_table}`, 'ml_generate_embedding_result', (SELECT ml_generate_embedding_result FROM ML.GENERATE_EMBEDDING(MODEL `{embedding_model}`, (SELECT \"{query_text}\" AS content))), top_k => {top_k})"""
    return bq_client.query(sql).to_dataframe()

def analyze_article_batch(df, disease, events, client, journal_dict):
    journal_context = "\n".join([f"- {title}: {sjr}" for title, sjr in journal_dict.items()])
    prompt = f"""Analyze articles for relevance to Disease: {disease} and Events: {', '.join(events)}. Use this data: {journal_context}. For each, extract: title, journal_title, journal_sjr, year, disease_match (bool), pediatric_focus (bool), treatment_shown (bool), paper_type, key_findings, clinical_trial (bool), novel_findings (bool). Return JSON array."""
    articles_text = ""
    for _, row in df.iterrows():
        content = row.get('content', row.get('abstract', ''))
        articles_text += f"\n---\nPMID: {row['PMID']}\nContent: {content}\n"
    response = client.models.generate_content(model=MODEL_ID, contents=[prompt + articles_text], config=GenerateContentConfig(temperature=0, response_mime_type="application/json"))
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        return []

def calculate_article_score(metadata, config):
    score = 0
    if metadata.get('disease_match'): score += config.get('disease_match', 50)
    if metadata.get('treatment_shown'): score += config.get('treatment_efficacy', 50)
    if metadata.get('clinical_trial'): score += config.get('clinical_trial', 40)
    return round(score, 2)

def normalize_journal_score(sjr):
    """Normalize journal SJR score using logarithmic scale (from gemini-medical-literature)."""
    if not sjr or sjr <= 0:
        return 0
    # Use log scale to handle large range of SJR values
    normalized = math.log(sjr + 1) * 5
    # Cap at 25 points
    return min(normalized, 25)

def calculate_dynamic_score(metadata, criteria_list, journal_dict):
    """Calculate article score based on dynamic criteria configuration."""
    score = 0
    current_year = datetime.now().year
    
    for criterion in criteria_list:
        # Skip if weight is 0
        if criterion['weight'] == 0:
            continue
            
        criterion_type = criterion.get('type', 'boolean')
        criterion_name = criterion['name']
        
        if criterion_type == 'special_journal':
            # Special handling for journal impact using SJR scores
            journal_title = metadata.get('journal_title', '')
            sjr = journal_dict.get(journal_title, 0)
            if sjr > 0:
                impact_score = normalize_journal_score(sjr)
                # Scale by user's weight (weight represents importance multiplier)
                score += impact_score * (criterion['weight'] / 25)  # Normalize to 25 max
                
        elif criterion_type == 'special_year':
            # Year penalty: -5 points per year from current
            if metadata.get('year'):
                try:
                    article_year = int(metadata.get('year'))
                    year_diff = current_year - article_year
                    year_penalty = -5 * year_diff
                    # Apply user's weight as a multiplier
                    score += year_penalty * criterion['weight']
                except (ValueError, TypeError):
                    pass
                    
        elif criterion_type == 'numeric':
            # For numeric criteria, multiply value by weight
            value = metadata.get(criterion_name, 0)
            if isinstance(value, (int, float)):
                score += value * criterion['weight']
                
        elif criterion_type == 'direct':
            # For direct scoring, use the value as-is (ignore weight)
            value = metadata.get(criterion_name, 0)
            if isinstance(value, (int, float)):
                score += value
                
        else:
            # Default: boolean criteria
            if metadata.get(criterion_name):
                score += criterion['weight']
                
    return round(score, 2)

def analyze_article_batch_with_criteria(df, disease, events, client, journal_dict, persona, criteria):
    """Analyze articles with custom persona and dynamic criteria."""
    # Build journal context
    journal_context = "\n".join([f"- {title}: {sjr}" for title, sjr in list(journal_dict.items())[:100]])  # Limit to first 100 for prompt size
    
    # Build criteria evaluation instructions dynamically from all criteria (except special ones)
    criteria_instructions = []
    for criterion in criteria:
        # Skip special criteria that are handled differently
        if criterion['name'] not in ['journal_impact', 'year']:
            if criterion['type'] == 'boolean':
                criteria_instructions.append(f"- {criterion['name']} (boolean): {criterion['description']}")
            elif criterion['type'] == 'numeric':
                criteria_instructions.append(f"- {criterion['name']} (number): {criterion['description']}")
            elif criterion['type'] == 'direct':
                criteria_instructions.append(f"- {criterion['name']} (number 0-100): {criterion['description']}")
    
    # Build the prompt with only standard fields hardcoded
    criteria_text = "\n".join(criteria_instructions) if criteria_instructions else ""
    
    prompt = f"""{persona}

Analyze the following articles for relevance to:
- Disease: {disease}
- Events: {', '.join(events)}

Journal Impact Data (sample):
{journal_context}

For each article, extract the following information:
1. Standard fields (always extract these):
   - title: Article title
   - journal_title: Name of the journal
   - journal_sjr: SJR score from the provided list (or 0 if not found)
   - year: Publication year

2. Evaluation criteria:
{criteria_text}

Return your analysis as a JSON array with one object per article.
"""
    
    # Compile articles text
    articles_text = ""
    for _, row in df.iterrows():
        content = row.get('content', row.get('abstract', ''))
        articles_text += f"\n---\nPMID: {row['PMID']}\nContent: {content}\n"
    
    # Generate analysis
    response = client.models.generate_content(
        model=MODEL_ID, 
        contents=[prompt + articles_text], 
        config=GenerateContentConfig(temperature=0, response_mime_type="application/json")
    )
    
    try:
        return json.loads(response.text)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON response: {response.text}")
        return []

def setup_bigquery(project, dataset, client, progress=gr.Progress()):
    """Setup BigQuery dataset and model with retry logic."""
    progress(0.8, desc="Setting up BigQuery dataset and model (may take a couple minutes if first time)...")
    
    # Create dataset if it doesn't exist
    try:
        client.get_dataset(f"{project}.{dataset}")
    except:
        client.create_dataset(bigquery.Dataset(f"{project}.{dataset}"), exists_ok=True)
    
    # Create model with retry logic
    model_query = f"CREATE MODEL IF NOT EXISTS `{project}.{dataset}.textembed` REMOTE WITH CONNECTION DEFAULT OPTIONS(endpoint='text-embedding-005');"
    
    max_retries = 3
    retry_delays = [5, 10, 15]
    
    for attempt in range(max_retries):
        try:
            print(f"Creating BigQuery embedding model (attempt {attempt + 1}/{max_retries})...")
            client.query(model_query).result()
            print(f"Successfully created BigQuery model for {project}.{dataset}")
            return f"✅ BigQuery setup complete for {project}.{dataset}"
            
        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed: {error_msg}")
            
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
                print(f"All {max_retries} attempts failed.")
                raise Exception(f"Failed to create BigQuery model after {max_retries} attempts. Last error: {error_msg}")

# --- Gradio App Logic ---
def get_initial_projects():
    """Get the list of projects for the dropdown."""
    projects = list_projects()
    if not projects:
        # Provide option to manually enter project ID
        return gr.update(choices=["[Enter Project ID Manually]"], value="[Enter Project ID Manually]"), "⚠️ Could not list projects automatically. You can either fix the authentication issue (see console) or enter your project ID manually in the field below."
    choices = [f"{p['name']} ({p['id']})" for p in projects]
    return gr.update(choices=choices, value=choices[0] if choices else None), f"✅ Found {len(projects)} projects. Select a project and click Proceed."

def proceed_with_project(project_selection, progress=gr.Progress()):
    """Check and set up the selected project, then move to the next tab."""
    global genai_client, bq_client, journal_impact_dict, PROJECT_ID, LOCATION, USER_DATASET
    if not project_selection:
        return "❌ Please select a project first.", gr.update(interactive=False), gr.update()

    project_id = project_selection.split('(')[-1].rstrip(')')
    PROJECT_ID = project_id
    
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

        # Use the shared setup logic
        genai_client, bq_client, journal_impact_dict = setup_project(PROJECT_ID, LOCATION, USER_DATASET, progress)

        status = f"✅ Setup complete for {PROJECT_ID}! You can now analyze a case."
        return status, gr.update(interactive=True), gr.update(selected=2)

    except Exception as e:
        return f"❌ Error: {e}", gr.update(interactive=False), gr.update()

def run_analysis(case_text, num_articles, progress=gr.Progress()):
    if not genai_client or not bq_client:
        return None, "❌ Please complete setup first.", {}, gr.update()
    progress(0.1, desc="Extracting medical info...")
    medical_info = extract_medical_info(case_text, genai_client)
    disease = medical_info.get('disease', '')
    events = [e.strip() for e in medical_info.get('events', '').split(',')]

    progress(0.3, desc="Searching PubMed...")
    embedding_model_path = f"{PROJECT_ID}.{USER_DATASET}.textembed"
    articles_df = search_pubmed_articles(disease, events, bq_client, embedding_model_path, PUBMED_TABLE, num_articles)

    progress(0.6, desc="Analyzing articles...")
    analyses = analyze_article_batch(articles_df, disease, events, genai_client, journal_impact_dict)

    for i, analysis in enumerate(analyses):
        for k, v in analysis.items():
            articles_df.loc[i, k] = v

    scoring_config = SCORING_PRESETS["Clinical Focus"]
    articles_df['score'] = articles_df.apply(lambda row: calculate_article_score(row, scoring_config), axis=1)
    articles_df = articles_df.sort_values('score', ascending=False).reset_index()

    progress(0.9, desc="Generating results...")
    results_table = articles_df[['score', 'title', 'journal_title', 'year']].head(10)
    results = {'articles': articles_df.to_dict('records'), 'disease': disease, 'events': events, 'case_text': case_text}
    return results_table, f"✅ Analysis complete for '{disease}'.", results, gr.update(selected=4)

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
                    
                    # Dynamic criteria display
                    criteria_display = gr.HTML(value="<div>Loading criteria...</div>")
                    
                    # Add controls
                    with gr.Row():
                        add_criterion_btn = gr.Button("➕ Add New Criterion", size="sm")
                        reset_criteria_btn = gr.Button("🔄 Reset to Defaults", size="sm")
                        
                    # Total weight display
                    total_weight_display = gr.Markdown("**Total Weight:** 0")
                    
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
                    
                    # Hidden components for interaction - create one for each possible criterion
                    weight_inputs = []
                    delete_inputs = []
                    for i in range(20):  # Support up to 20 criteria
                        weight_inputs.append(gr.Textbox(visible=False, elem_id=f"weight-input-{i}"))
                        delete_inputs.append(gr.Textbox(visible=False, elem_id=f"delete-input-{i}"))
                    
                analyze_btn = gr.Button("Run Full Analysis", variant="primary", interactive=False)
                analysis_status = gr.Markdown()

            with gr.TabItem("4. Results", id=4):
                results_df = gr.DataFrame(label="Top 10 Ranked Articles")

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

        def handle_project_creation(project_id, billing_account, progress=gr.Progress()):
            status, new_project_selection = create_new_project(project_id, billing_account, progress)
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
            inputs=[new_project_id_input, billing_account_dropdown], 
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

        def handle_link_billing(billing_account, project_dropdown, progress=gr.Progress()):
            """Handle linking billing account to the project."""
            if not billing_account or billing_account == CREATE_BILLING_ACCOUNT_OPTION:
                return "❌ Please select a valid billing account.", gr.update(visible=True), gr.update(visible=False)
            
            project_id = project_dropdown.split('(')[-1].rstrip(')')
            progress(0.1, desc="Linking billing account...")
            
            success, message = link_billing_to_project(project_id, billing_account)
            if success:
                progress(0.3, desc="Billing linked! Continuing setup...")
                # After successful billing link, continue with the normal setup
                status, analyze_btn_update, tabs_update = proceed_with_project(project_dropdown, progress)
                # Return appropriate updates for this function's outputs
                # The .then() chains will handle the analyze button and tabs updates based on the status message
                return status, gr.update(visible=False), gr.update(visible=True)
            else:
                return message, gr.update(visible=True), gr.update(visible=False)

        # State to track if we need billing setup
        needs_billing_setup = gr.State(False)
        
        # Modified proceed button click handler
        def handle_proceed_click(project_dropdown, progress=gr.Progress()):
            """Handle the proceed button click."""
            status, analyze_btn_update, tabs_update = proceed_with_project(project_dropdown, progress)
            
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
            inputs=[project_dropdown], 
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
            inputs=[billing_setup_dropdown, project_dropdown],
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
        # Enable/disable proceed button based on case input
        def check_case_input(case_text):
            if case_text.strip():
                return gr.update(interactive=True)
            else:
                return gr.update(interactive=False)
        
        case_input.change(
            check_case_input,
            inputs=[case_input],
            outputs=[proceed_to_persona_btn]
        )
        
        # Load example button handler
        def load_example_case():
            return SAMPLE_CASE, gr.update(interactive=True)

        load_example_btn.click(
            load_example_case,
            outputs=[case_input, proceed_to_persona_btn]
        )

        # Modified proceed button handler to go to Persona tab
        def proceed_to_persona(case_text):
            if not case_text.strip():
                return "❌ Please enter case notes first.", gr.update(interactive=False), gr.update()
            return "✅ Case notes saved. Please customize your persona.", gr.update(interactive=True), gr.update(selected=3)

        proceed_to_persona_btn.click(
            proceed_to_persona,
            inputs=[case_input],
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
            {"name": "disease_match", "description": "Does the article match the patient's disease?", "weight": 50, "type": "boolean", "deletable": True},
            {"name": "treatment_shown", "description": "Does the article show positive treatment results?", "weight": 50, "type": "boolean", "deletable": True},
            {"name": "pediatric_focus", "description": "Does the article focus on pediatric patients?", "weight": 20, "type": "boolean", "deletable": True},
            {"name": "clinical_trial", "description": "Is this a clinical trial?", "weight": 40, "type": "boolean", "deletable": True},
            {"name": "novelty", "description": "Does the article present novel findings?", "weight": 10, "type": "boolean", "deletable": True},
            {"name": "actionable_events_match", "description": "How many actionable events from the patient's case are mentioned in this article?", "weight": 15, "type": "direct", "deletable": True},
            {"name": "human_clinical_data", "description": "Does the article include human clinical data?", "weight": 15, "type": "boolean", "deletable": True},
            {"name": "cell_studies", "description": "Does the article include cell studies?", "weight": 5, "type": "boolean", "deletable": True},
            {"name": "mice_studies", "description": "Does the article include mice studies?", "weight": 10, "type": "boolean", "deletable": True},
            {"name": "journal_impact", "description": "Journal impact factor (SJR)", "weight": 25, "type": "special_journal", "deletable": True},
            {"name": "year", "description": "Publication year penalty", "weight": 1, "type": "special_year", "deletable": True}
        ]
        
        def render_criteria_html(criteria_list):
            """Render criteria as HTML for display."""
            html = '<div style="display: flex; flex-direction: column; gap: 12px;">'
            
            for idx, criterion in enumerate(criteria_list):
                deletable = criterion.get('deletable', True)
                user_defined = criterion.get('user_defined', False)
                
                # Different colors for different types
                type_colors = {
                    'boolean': '#4CAF50',
                    'numeric': '#2196F3', 
                    'direct': '#FF9800',
                    'special_journal': '#9C27B0',
                    'special_year': '#F44336'
                }
                type_color = type_colors.get(criterion['type'], '#607D8B')
                
                html += f'''
                <div style="border: 1px solid #d0d5dd; border-radius: 8px; padding: 15px; background: #e8eaf0;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                        <div style="flex: 1;">
                            <strong style="font-size: 16px; color: #1f2937;">{criterion['description']}</strong>
                            <div style="margin-top: 5px;">
                                <span style="background: {type_color}; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; margin-right: 8px;">
                                    {criterion['type']}
                                </span>
                                <span style="color: #666; font-size: 12px;">Name: {criterion['name']}</span>
                            </div>
                        </div>
                    </div>
                    <div style="display: flex; align-items: center; gap: 16px;">
                        <div style="flex: 1;">
                            <label style="font-size: 14px; color: #666;">Weight: <strong id="weight-{idx}">{criterion['weight']}</strong></label>
                            <input type="range" min="0" max="100" value="{criterion['weight']}" 
                                   style="width: 100%; margin-top: 5px;"
                                   oninput="document.getElementById('weight-{idx}').textContent = this.value; 
                                            document.getElementById('weight-input-{idx}').value = this.value;
                                            document.getElementById('weight-input-{idx}').dispatchEvent(new Event('input'));"
                                   id="slider-{idx}">
                        </div>
                        <button style="background: {'#9ca3af' if not deletable else '#ef4444'}; color: white; border: none; 
                                       padding: 6px 12px; border-radius: 4px; cursor: {'not-allowed' if not deletable else 'pointer'}; 
                                       font-size: 14px;"
                                onclick="if ({str(deletable).lower()}) {{ 
                                    document.getElementById('delete-input-{idx}').value = 'delete'; 
                                    document.getElementById('delete-input-{idx}').dispatchEvent(new Event('input')); 
                                }}"
                                {'disabled' if not deletable else ''}>
                            🗑️ Delete
                        </button>
                    </div>
                </div>
                '''
            
            html += '</div>'
            
            # Add hidden inputs for each criterion
            html += '<div style="display: none;">'
            for idx in range(len(criteria_list)):
                html += f'<input type="text" id="weight-input-{idx}" value="">'
                html += f'<input type="text" id="delete-input-{idx}" value="">'
            html += '</div>'
            
            return html
        
        def calculate_total_weight(criteria_list):
            """Calculate total weight from criteria list."""
            total = sum(c['weight'] for c in criteria_list)
            return f"**Total Weight:** {total}"
        
        def update_criterion_weight(criteria_list, idx, new_weight):
            """Update weight for a specific criterion."""
            if new_weight and 0 <= idx < len(criteria_list):
                try:
                    criteria_list[idx]['weight'] = int(new_weight)
                except ValueError:
                    pass
            return criteria_list, render_criteria_html(criteria_list), calculate_total_weight(criteria_list)
        
        def delete_criterion(criteria_list, idx):
            """Delete a criterion from the list."""
            if 0 <= idx < len(criteria_list) and criteria_list[idx].get('deletable', True):
                criteria_list.pop(idx)
            return criteria_list, render_criteria_html(criteria_list), calculate_total_weight(criteria_list)
        
        def show_add_criterion_dialog():
            """Show the dialog for adding a new criterion."""
            return gr.update(visible=True)
        
        def hide_add_criterion_dialog():
            """Hide the add criterion dialog."""
            return gr.update(visible=False), "", "boolean", 10
        
        def add_new_criterion_with_details(criteria_list, description, criterion_type, weight):
            """Add a new custom criterion with user-provided details."""
            if not description.strip():
                return criteria_list, render_criteria_html(criteria_list), calculate_total_weight(criteria_list), gr.update(visible=True)
            
            # Generate a safe name from description
            import re
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', description.lower())[:30]
            if not safe_name or safe_name[0].isdigit():
                safe_name = f"custom_{len(criteria_list)}"
            
            new_criterion = {
                "name": safe_name,
                "description": description.strip(),
                "weight": weight,
                "type": criterion_type,
                "deletable": True,
                "user_defined": True
            }
            criteria_list.append(new_criterion)
            return criteria_list, render_criteria_html(criteria_list), calculate_total_weight(criteria_list), gr.update(visible=False)
        
        def reset_to_defaults():
            """Reset criteria to defaults."""
            return DEFAULT_CRITERIA.copy(), render_criteria_html(DEFAULT_CRITERIA), calculate_total_weight(DEFAULT_CRITERIA)
        
        # Initialize criteria state on load
        def initialize_criteria():
            criteria = DEFAULT_CRITERIA.copy()
            return criteria, render_criteria_html(criteria), calculate_total_weight(criteria)
        
        # Set up initial criteria
        demo.load(
            initialize_criteria,
            outputs=[criteria_state, criteria_display, total_weight_display]
        )
        
        # Criteria management event handlers
        add_criterion_btn.click(
            show_add_criterion_dialog,
            outputs=[add_criterion_dialog]
        )
        
        cancel_add_btn.click(
            hide_add_criterion_dialog,
            outputs=[add_criterion_dialog, new_criterion_description, new_criterion_type, new_criterion_weight]
        )
        
        confirm_add_btn.click(
            add_new_criterion_with_details,
            inputs=[criteria_state, new_criterion_description, new_criterion_type, new_criterion_weight],
            outputs=[criteria_state, criteria_display, total_weight_display, add_criterion_dialog]
        ).then(
            lambda: ("", "boolean", 10),
            outputs=[new_criterion_description, new_criterion_type, new_criterion_weight]
        )
        
        reset_criteria_btn.click(
            reset_to_defaults,
            outputs=[criteria_state, criteria_display, total_weight_display]
        )
        
        # Wire up weight and delete handlers for each input
        for i in range(20):
            weight_inputs[i].change(
                lambda weight, idx=i: update_criterion_weight(criteria_state.value, idx, weight) if criteria_state.value else (criteria_state.value, render_criteria_html(criteria_state.value), calculate_total_weight(criteria_state.value)),
                inputs=[weight_inputs[i]],
                outputs=[criteria_state, criteria_display, total_weight_display]
            )
            
            delete_inputs[i].change(
                lambda action, idx=i: delete_criterion(criteria_state.value, idx) if action == "delete" and criteria_state.value else (criteria_state.value, render_criteria_html(criteria_state.value), calculate_total_weight(criteria_state.value)),
                inputs=[delete_inputs[i]],
                outputs=[criteria_state, criteria_display, total_weight_display]
            ).then(
                lambda: "",
                outputs=[delete_inputs[i]]
            )
        
        # Modified analyze button to use persona and criteria
        def run_analysis_with_persona(case_text, num_articles, persona, criteria, progress=gr.Progress()):
            """Run analysis with custom persona and scoring criteria."""
            if not genai_client or not bq_client:
                return None, "❌ Please complete setup first.", {}, gr.update()
            
            progress(0.1, desc="Extracting medical info...")
            medical_info = extract_medical_info(case_text, genai_client)
            disease = medical_info.get('disease', '')
            events = [e.strip() for e in medical_info.get('events', '').split(',')]

            progress(0.3, desc="Searching PubMed...")
            embedding_model_path = f"{PROJECT_ID}.{USER_DATASET}.textembed"
            articles_df = search_pubmed_articles(disease, events, bq_client, embedding_model_path, PUBMED_TABLE, num_articles)

            progress(0.6, desc="Analyzing articles with custom criteria...")
            # Modified to pass persona and criteria
            analyses = analyze_article_batch_with_criteria(articles_df, disease, events, genai_client, journal_impact_dict, persona, criteria)

            for i, analysis in enumerate(analyses):
                for k, v in analysis.items():
                    articles_df.loc[i, k] = v

            # Use dynamic scoring
            articles_df['score'] = articles_df.apply(
                lambda row: calculate_dynamic_score(row, criteria, journal_impact_dict), 
                axis=1
            )
            articles_df = articles_df.sort_values('score', ascending=False).reset_index()

            progress(0.9, desc="Generating results...")
            results_table = articles_df[['score', 'title', 'journal_title', 'year']].head(10)
            results = {
                'articles': articles_df.to_dict('records'), 
                'disease': disease, 
                'events': events, 
                'case_text': case_text,
                'persona': persona,
                'criteria': criteria
            }
            return results_table, f"✅ Analysis complete for '{disease}'.", results, gr.update(selected=4)
        
        analyze_btn.click(
            run_analysis_with_persona,
            inputs=[case_input, num_articles_slider, persona_text, criteria_state],
            outputs=[results_df, analysis_status, app_state, tabs]
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
