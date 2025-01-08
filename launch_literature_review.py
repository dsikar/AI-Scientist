import argparse
import json
import multiprocessing
import os
import os.path as osp
import shutil
import sys
import time
from datetime import datetime

from ai_scientist.llm import create_client, AVAILABLE_LLMS
from ai_scientist.perform_literature_review import perform_literature_review
from ai_scientist.paper_loader import load_paper_text

def print_time():
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

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
    parser.add_argument(
        "--parallel",
        type=int,
        default=0,
        help="Number of parallel processes to run. 0 for sequential execution."
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Skip paper search and use existing papers"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true", 
        help="Skip paper analysis and use existing analyses"
    )
    return parser.parse_args()

def worker(queue, output_dir, client, client_model, temperature):
    print(f"Worker started")
    while True:
        paper = queue.get()
        if paper is None:
            break
        try:
            paper_dir = osp.join(output_dir, f"{paper['paperId']}")
            os.makedirs(paper_dir, exist_ok=True)
            
            if os.path.exists(osp.join(paper_dir, "analysis.json")):
                print(f"Analysis exists for {paper['title']}, skipping")
                continue
                
            # Download and analyze paper
            pdf_path = osp.join(paper_dir, "paper.pdf")
            # TODO: Implement paper download
            paper_text = load_paper_text(pdf_path)
            
            analysis = perform_literature_review(
                paper_text,
                output_dir=paper_dir,
                client=client,
                model=client_model,
                temperature=temperature
            )
            
            with open(osp.join(paper_dir, "analysis.json"), "w") as f:
                json.dump(analysis, f, indent=2)
                
            print(f"Completed analysis of: {paper['title']}")
            
        except Exception as e:
            print(f"Failed to analyze paper {paper['title']}: {str(e)}")
    print(f"Worker finished")

if __name__ == "__main__":
    args = parse_arguments()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = osp.join("templates", "literature_review")
    output_dir = osp.join("results", "literature_reviews", f"{timestamp}_{args.topic}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy template files
    shutil.copytree(base_dir, output_dir, dirs_exist_ok=True)
    
    # Create LLM client
    client, client_model = create_client(args.model)
    
    print_time()
    print("Starting literature review")
    
    # Load or search for papers
    papers_file = osp.join(output_dir, "papers.json")
    if args.skip_search and os.path.exists(papers_file):
        with open(papers_file, "r") as f:
            papers = json.load(f)
        print(f"Loaded {len(papers)} existing papers")
    else:
        papers = search_for_papers(
            args.topic,
            max_papers=args.max_papers,
            min_citations=args.min_citations
        )
        with open(papers_file, "w") as f:
            json.dump(papers, f, indent=2)
        print(f"Found {len(papers)} papers")
    
    if args.parallel > 0:
        print(f"Running {args.parallel} parallel processes")
        queue = multiprocessing.Queue()
        for paper in papers:
            queue.put(paper)
            
        processes = []
        for _ in range(args.parallel):
            p = multiprocessing.Process(
                target=worker,
                args=(queue, output_dir, client, client_model, args.temperature)
            )
            p.start()
            processes.append(p)
            
        # Signal workers to exit
        for _ in range(args.parallel):
            queue.put(None)
            
        for p in processes:
            p.join()
            
        print("All parallel processes completed")
    else:
        for paper in papers:
            try:
                paper_dir = osp.join(output_dir, f"{paper['paperId']}")
                os.makedirs(paper_dir, exist_ok=True)
                
                if os.path.exists(osp.join(paper_dir, "analysis.json")):
                    print(f"Analysis exists for {paper['title']}, skipping")
                    continue
                    
                # Download and analyze paper
                pdf_path = osp.join(paper_dir, "paper.pdf")
                # TODO: Implement paper download
                paper_text = load_paper_text(pdf_path)
                
                analysis = perform_literature_review(
                    paper_text,
                    output_dir=paper_dir,
                    client=client,
                    model=client_model,
                    temperature=args.temperature
                )
                
                with open(osp.join(paper_dir, "analysis.json"), "w") as f:
                    json.dump(analysis, f, indent=2)
                    
                print(f"Completed analysis of: {paper['title']}")
                
            except Exception as e:
                print(f"Failed to analyze paper {paper['title']}: {str(e)}")
                
    print_time()
    print("Literature review completed")
    
    # Synthesize results if analyses exist
    analyses = []
    for paper in papers:
        analysis_file = osp.join(output_dir, paper['paperId'], "analysis.json")
        if os.path.exists(analysis_file):
            with open(analysis_file, "r") as f:
                analyses.append(json.load(f))
                
    if analyses:
        synthesis = synthesize_analyses(
            analyses,
            topic=args.topic,
            client=client,
            model=client_model,
            temperature=args.temperature
        )
        
        with open(osp.join(output_dir, "synthesis.json"), "w") as f:
            json.dump(synthesis, f, indent=2)
            
        print(f"Literature review completed successfully. Results saved to {output_dir}")
    else:
        print("No analyses found to synthesize")
