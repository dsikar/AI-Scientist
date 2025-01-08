import json
import os
import os.path as osp
from typing import List, Dict, Optional, Tuple

from ai_scientist.llm import get_response_from_llm, extract_json_between_markers
from ai_scientist.generate_ideas import search_for_papers
from ai_scientist.paper_loader import load_paper_text
from ai_scientist.paper_downloader import download_paper, extract_metadata, PaperAccessError

literature_system_msg = """You are an AI research assistant helping to perform a comprehensive literature review.
Your goal is to analyze academic papers and synthesize their key findings, methods, and contributions.
Be thorough and analytical in your assessment."""

paper_analysis_prompt = """Please analyze the following paper and extract key information.

Paper text:
```
{text}
```

Respond in the following format:

THOUGHT:
<THOUGHT>

ANALYSIS JSON:
```json
<JSON>
```

In <THOUGHT>, briefly discuss your analysis approach and key insights from the paper.

In <JSON>, provide the analysis in JSON format with the following fields:
- "Title": The paper's title
- "Authors": List of authors
- "Year": Publication year
- "Venue": Publication venue
- "Problem": The main problem/challenge addressed
- "Methods": Key methods and techniques used
- "Results": Main results and findings
- "Contributions": List of key contributions
- "Limitations": List of limitations and constraints
- "Future_Work": Suggested future work
- "Related_Papers": List of important cited works to follow up on
- "Keywords": List of key topics/terms
- "Rating": Rating from 1-10 on relevance to the review topic
"""

synthesis_prompt = """You have analyzed {num_papers} papers related to {topic}. Here are the analyses:

{analyses}

Please synthesize these papers into a coherent literature review.

Respond in the following format:

THOUGHT:
<THOUGHT>

SYNTHESIS JSON:
```json
<JSON>
```

In <THOUGHT>, discuss your approach to synthesizing these papers and key themes/patterns you identified.

In <JSON>, provide the synthesis in JSON format with the following fields:
- "Overview": High-level summary of the research area
- "Key_Themes": List of major themes/topics identified
- "Methods_Summary": Overview of common methods/approaches
- "Results_Summary": Summary of key findings across papers
- "Open_Problems": List of open challenges and problems
- "Future_Directions": Promising future research directions
- "Taxonomy": Categorization/grouping of the papers
- "Timeline": Timeline of key developments
- "Recommendations": Suggested next steps for researchers
"""

def analyze_paper(
    paper_text: str,
    client,
    model: str,
    topic: str,
    temperature: float = 0.7
) -> Optional[Dict]:
    """Analyze a single academic paper using LLM."""
    try:
        text, msg_history = get_response_from_llm(
            paper_analysis_prompt.format(text=paper_text),
            client=client,
            model=model,
            system_message=literature_system_msg,
            temperature=temperature
        )
        analysis = extract_json_between_markers(text)
        return analysis
    except Exception as e:
        print(f"Failed to analyze paper: {e}")
        return None

def synthesize_analyses(
    analyses: List[Dict],
    topic: str,
    client,
    model: str,
    temperature: float = 0.7
) -> Optional[Dict]:
    """Synthesize multiple paper analyses into a literature review."""
    try:
        analyses_text = json.dumps(analyses, indent=2)
        text, msg_history = get_response_from_llm(
            synthesis_prompt.format(
                num_papers=len(analyses),
                topic=topic,
                analyses=analyses_text
            ),
            client=client,
            model=model,
            system_message=literature_system_msg,
            temperature=temperature
        )
        synthesis = extract_json_between_markers(text)
        return synthesis
    except Exception as e:
        print(f"Failed to synthesize analyses: {e}")
        return None

def perform_literature_review(
    topic: str,
    output_dir: str,
    client,
    model: str,
    max_papers: int = 20,
    min_citations: int = 10,
    temperature: float = 0.7
) -> Optional[Dict]:
    """Perform automated literature review on a topic."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Search for relevant papers
    papers = search_for_papers(topic, result_limit=max_papers)
    if not papers:
        print(f"No papers found for topic: {topic}")
        return None
        
    # Filter by citation count
    papers = [p for p in papers if p.get("citationCount", 0) >= min_citations]
    
    # Get absolute path for output directory
    abs_output_dir = os.path.abspath(output_dir)
    
    # Create papers directory
    papers_dir = os.path.join(abs_output_dir, "papers")
    os.makedirs(papers_dir, exist_ok=True)
    
    # Analyze each paper
    analyses = []
    for paper in papers:
        paper_id = paper.get("paperId")
        if not paper_id:
            continue
            
        paper_dir = os.path.join(papers_dir, paper_id)
        os.makedirs(paper_dir, exist_ok=True)
        
        # Try to get full text first
        try:
            pdf_path = download_paper(paper, paper_dir)
            paper_text = load_paper_text(pdf_path)
            analysis = analyze_paper(paper_text, client, model, topic, temperature)
            
        except (PaperAccessError, Exception) as e:
            print(f"Failed to get full text for {paper_id}, falling back to metadata: {e}")
            
            # Extract available metadata
            metadata = extract_metadata(paper)
            
            # Save metadata
            metadata_path = os.path.join(paper_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            # Analyze with limited info
            metadata_text = f"""Title: {metadata['title']}
Authors: {', '.join(metadata['authors'])}
Year: {metadata['year']}
Venue: {metadata['venue']}
Citations: {metadata['citation_count']}
Abstract: {metadata['abstract']}"""

            analysis = analyze_paper(metadata_text, client, model, topic, temperature)
            
        if analysis:
            # Add metadata about paper access
            analysis["pdf_status"] = paper.get("openAccessPdf", {}).get("status")
            analysis["analysis_type"] = "full_text" if "pdf_path" in locals() else "metadata_only"
            analyses.append(analysis)
            
    if not analyses:
        print("No papers could be analyzed successfully")
        return None
        
    # Synthesize analyses into review
    synthesis = synthesize_analyses(analyses, topic, client, model, temperature)
    
    # Save results
    analyses_path = os.path.join(abs_output_dir, "analyses.json")
    synthesis_path = os.path.join(abs_output_dir, "synthesis.json")
    
    with open(analyses_path, "w") as f:
        json.dump(analyses, f, indent=2)
    with open(synthesis_path, "w") as f:
        json.dump(synthesis, f, indent=2)
        
    return synthesis
