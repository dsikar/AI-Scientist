import json
import time
import requests
from typing import Optional, List, Dict

def search_semantic_scholar(
    query: str,
    result_limit: int = 10
) -> Optional[List[Dict]]:
    """Search for papers using Semantic Scholar API and print detailed response info."""
    if not query:
        return None
        
    print(f"\nSearching Semantic Scholar for: {query}")
    print(f"Result limit: {result_limit}")
    
    try:
        # Add delay to respect rate limits
        time.sleep(5)
        
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": result_limit,
            "fields": "title,authors,venue,year,abstract,citationStyles,citationCount,url,openAccessPdf"
        }
        
        print("\nMaking API request to:", url)
        print("With parameters:", json.dumps(params, indent=2))
        
        rsp = requests.get(url, params=params, timeout=30)
        print("\nResponse status code:", rsp.status_code)
        print("Response headers:", json.dumps(dict(rsp.headers), indent=2))
        
        if rsp.status_code != 200:
            print("Error response:", rsp.text)
            return None
            
        results = rsp.json()
        total = results.get("total", 0)
        papers = results.get("data", [])
        
        print(f"\nFound {total} total results")
        print(f"Returned {len(papers)} papers in this response")
        
        if papers:
            print("\nPaper details:")
            for i, paper in enumerate(papers, 1):
                print(f"\n--- Paper {i} ---")
                print("All available fields:")
                for key, value in paper.items():
                    print(f"{key}: {value}")
                
                # If there's an openAccessPdf field, try to verify the URL
                if paper.get('openAccessPdf'):
                    pdf_info = paper['openAccessPdf']
                    print("\nTesting PDF download:")
                    print(f"Status: {pdf_info.get('status')}")
                    if pdf_url := pdf_info.get('url'):
                        try:
                            # Just do a HEAD request to check content type
                            head_resp = requests.head(pdf_url, timeout=30)
                            print(f"Content-Type: {head_resp.headers.get('Content-Type')}")
                            print(f"Content-Length: {head_resp.headers.get('Content-Length')}")
                            print(f"Status Code: {head_resp.status_code}")
                        except requests.exceptions.RequestException as e:
                            print(f"Failed to verify PDF URL: {e}")
                
            return papers
            
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"\nError searching for papers: {e}")
        return None

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Debug Semantic Scholar API search")
    parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of results"
    )
    
    args = parser.parse_args()
    papers = search_semantic_scholar(args.query, args.limit)
