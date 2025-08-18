# This file contains the code for final analysis cells to be added to the notebook
# Copy each section into a new cell in your Jupyter notebook

# ============= MARKDOWN CELL 1 =============
"""
### 12. Final Analysis with Custom Prompt Templates

Generate comprehensive literature synthesis with customizable analysis prompts.
"""

# ============= CODE CELL 1 =============
# Default final analysis prompt template
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

# ============= CODE CELL 2 =============
def format_article_for_analysis(article, idx):
    """Format a single article for the analysis prompt."""
    metadata = article.get('metadata', article)
    
    # Get events found
    events_found = metadata.get('actionable_events', 'None')
    if isinstance(events_found, str) and events_found:
        events_str = events_found
    else:
        events_str = "None identified"
    
    # Handle journal info - try different fields
    journal = metadata.get('journal_title', metadata.get('journal', 'Unknown'))
    
    return f"""
Article {idx}:
Title: {metadata.get('title', 'Unknown')}
Journal: {journal} | Year: {metadata.get('year', 'N/A')}
Type: {metadata.get('paper_type', 'Unknown')}
Score: {article.get('score', 0):.1f}
Key Concepts Found: {events_str}
PMID: {article.get('pmid', 'N/A')} | PMCID: {article.get('pmcid', 'N/A')}

Full Text:
{article.get('content', 'No content available')[:5000]}...
"""

def create_final_analysis_prompt(case_text, disease, events, articles, custom_template=None):
    """Create the final analysis prompt with full article contents."""
    
    if not articles:
        return None
    
    # Use custom template or default
    template = custom_template or FINAL_ANALYSIS_PROMPT_TEMPLATE
    
    # Format all articles
    articles_content_parts = []
    for idx, article in enumerate(articles, 1):
        articles_content_parts.append(format_article_for_analysis(article, idx))
    
    # Join all articles with separator
    articles_content = ("\n" + "="*80 + "\n").join(articles_content_parts)
    
    # Fill in the template
    filled_prompt = template.format(
        case_description=case_text,
        primary_focus=disease,
        key_concepts=', '.join(events),
        articles_content=articles_content
    )
    
    return filled_prompt

# ============= CODE CELL 3 =============
def generate_final_analysis(results, articles_to_analyze=None, custom_template=None):
    """Generate comprehensive final analysis of the literature."""
    
    # Use provided articles or top 10
    if articles_to_analyze is None:
        articles_to_analyze = results['articles'].head(10).to_dict('records')
    
    if not articles_to_analyze:
        return "❌ No articles available for analysis."
    
    print(f"🔄 Generating final analysis for {len(articles_to_analyze)} articles...")
    
    # Create the prompt
    prompt = create_final_analysis_prompt(
        results['case_text'],
        results['disease'],
        results['events'],
        articles_to_analyze,
        custom_template
    )
    
    if not prompt:
        return "❌ Could not create analysis prompt."
    
    # Generate analysis
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=[prompt],
        config=GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=THINKING_BUDGET)
        )
    )
    
    return response.text

# ============= CODE CELL 4 =============
# Interactive article selection widget
class ArticleSelector:
    """Widget for selecting articles to include in final analysis."""
    
    def __init__(self, results):
        self.results = results
        self.articles_df = results['articles']
        self.selected_indices = []
        
        # Create checkboxes for each article
        self.checkboxes = []
        for idx, (_, article) in enumerate(self.articles_df.iterrows()):
            # Handle journal info - try different fields
            journal = article.get('journal_title', article.get('journal', 'Unknown'))
            
            label = f"[Score: {article['score']:.1f}] {article.get('title', 'Unknown')[:80]}... ({journal}, {article.get('year', 'N/A')})"
            checkbox = widgets.Checkbox(
                value=idx < 5,  # Select top 5 by default
                description=label,
                layout=widgets.Layout(width='100%'),
                style={'description_width': 'initial'}
            )
            self.checkboxes.append(checkbox)
        
        # Select/deselect all buttons
        self.select_all_btn = widgets.Button(
            description='Select All',
            button_style='success',
            icon='check-square'
        )
        self.deselect_all_btn = widgets.Button(
            description='Deselect All',
            button_style='warning',
            icon='square'
        )
        
        self.select_all_btn.on_click(lambda b: self._select_all())
        self.deselect_all_btn.on_click(lambda b: self._deselect_all())
        
        # Custom prompt template
        self.custom_prompt = widgets.Textarea(
            value=FINAL_ANALYSIS_PROMPT_TEMPLATE,
            placeholder='Enter custom analysis prompt template...',
            description='Analysis Prompt:',
            layout=widgets.Layout(width='100%', height='200px'),
            style={'description_width': 'initial'}
        )
        
        # Generate button
        self.generate_btn = widgets.Button(
            description='Generate Final Analysis',
            button_style='primary',
            icon='play'
        )
        self.generate_btn.on_click(lambda b: self._generate_analysis())
        
        # Output area
        self.output = widgets.Output()
        
    def _select_all(self):
        for checkbox in self.checkboxes:
            checkbox.value = True
            
    def _deselect_all(self):
        for checkbox in self.checkboxes:
            checkbox.value = False
            
    def _get_selected_articles(self):
        selected = []
        for idx, checkbox in enumerate(self.checkboxes):
            if checkbox.value:
                selected.append(self.articles_df.iloc[idx].to_dict())
        return selected
    
    def _generate_analysis(self):
        selected_articles = self._get_selected_articles()
        
        with self.output:
            clear_output(wait=True)
            
            if not selected_articles:
                display(HTML('<p style="color: red;">❌ Please select at least one article.</p>'))
                return
            
            display(HTML(f'<p>🔄 Generating analysis for {len(selected_articles)} selected articles...</p>'))
            
            # Generate analysis
            analysis = generate_final_analysis(
                self.results,
                selected_articles,
                self.custom_prompt.value
            )
            
            clear_output(wait=True)
            display(HTML('<h3>📊 Final Literature Analysis</h3>'))
            display(Markdown(analysis))
    
    def display(self):
        """Display the article selector interface."""
        display(widgets.VBox([
            widgets.HTML('<h3>📚 Select Articles for Final Analysis</h3>'),
            widgets.HBox([self.select_all_btn, self.deselect_all_btn]),
            widgets.HTML('<hr>'),
            *self.checkboxes,
            widgets.HTML('<hr>'),
            widgets.HTML('<h4>Customize Analysis Prompt (Optional)</h4>'),
            self.custom_prompt,
            widgets.HTML('<hr>'),
            self.generate_btn,
            self.output
        ]))

# ============= MARKDOWN CELL 2 =============
"""
### Step 5: Generate Final Literature Analysis

Select articles and customize the analysis prompt to generate a comprehensive literature synthesis.
"""

# ============= CODE CELL 5 (To use in the example) =============
# Create article selector for final analysis
article_selector = ArticleSelector(results)
article_selector.display()

# ============= CODE CELL 6 (Alternative simple usage) =============
# Or generate final analysis directly with top 10 articles
# final_analysis = generate_final_analysis(results)
# display(HTML('<h3>📊 Final Literature Analysis</h3>'))
# display(Markdown(final_analysis))
