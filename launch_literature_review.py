import argparse
import json
import os.path as osp
from datetime import datetime

from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_literature_review import perform_literature_review

def parse_arguments():
    parser = argparse.ArgumentParser(description="Perform automated literature review")
    parser.add_argument(
        "topic",
        type=str,
        help="Research topic to review"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for LLM sampling"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = osp.join("results", "literature_reviews", f"{timestamp}_{args.topic}")
    
    # Create LLM client
    client, client_model = create_client(args.model)
    
    # Perform review
    synthesis = perform_literature_review(
        topic=args.topic,
        output_dir=output_dir,
        client=client,
        model=client_model,
        max_papers=args.max_papers,
        min_citations=args.min_citations,
        temperature=args.temperature
    )
    
    if synthesis:
        print(f"Literature review completed successfully. Results saved to {output_dir}")
    else:
        print("Literature review failed")
