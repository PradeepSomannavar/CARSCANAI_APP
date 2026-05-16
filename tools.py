import os
import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from dotenv import load_dotenv
from pathlib import Path
from langchain_core.tools import tool

# Load env from current directory
load_dotenv(Path(__file__).parent / ".env")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

@tool
def web_search(query: str) -> str:
    """Search the web using Tavily API for repair costs, part prices, and automotive data. Pass a specific search query."""
    if not TAVILY_API_KEY:
        return "Error: TAVILY_API_KEY not found in environment."
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query=query, search_depth="advanced", max_results=5)
        
        results = []
        for result in response.get('results', []):
            results.append(f"Source: {result['url']}\nContent: {result['content']}\n")
        
        return "\n---\n".join(results) if results else "No relevant search results found."
    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def scrape_url(url: str) -> str:
    """Scrape text content from a URL to extract pricing data, part costs, and detailed information. Must be a valid HTTP/HTTPS URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        for script in soup(["script", "style"]):
            script.extract()
            
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text[:5000]
    except Exception as e:
        return f"Scrape error for {url}: {str(e)}"
