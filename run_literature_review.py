import argparse
import json
import os
import os.path as osp
from datetime import datetime

from ai_scientist.llm import create_client
from ai_scientist.perform_literature_review import perform_literature_review

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI Agents literature review")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4-turbo-preview",
        help="Model to use for analysis"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        default=20,
        help="Maximum number of papers to analyze"
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=10,
        help="Minimum citation count for papers"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Load search keywords
    with open("templates/literature_review/search_keywords.json", "r") as f:
        search_config = json.load(f)
        
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = osp.join("results", "literature_reviews", f"{timestamp}_ai_agents")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LLM client
    client, client_model = create_client(args.model)
    
    # Run literature review
    print(f"Starting literature review on {search_config['primary_query']}")
    synthesis = perform_literature_review(
        topic=search_config['primary_query'],
        output_dir=output_dir,
        client=client,
        model=client_model,
        max_papers=args.max_papers,
        min_citations=args.min_citations
    )
    
    if synthesis:
        print(f"Literature review completed successfully. Results saved to {output_dir}")
    else:
        print("Literature review failed")
