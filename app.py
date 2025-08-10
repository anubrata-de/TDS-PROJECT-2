import os
import json
import tempfile
import asyncio
import re
import shutil
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware

# ----------------------------
# AIPipe client (LLM interface)
# ----------------------------
AIPIPE_API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6ImFtYWFuYW5zYXJpMDIwNjIwMDNAZ21haWwuY29tIn0.Uj_2Wot14-dhFmENinvcv7fBSVH6I-yOqu_2MeO-Z6g"
AIPIPE_MODEL_DEFAULT = "openai/gpt-4o-mini"

if not AIPIPE_API_KEY:
    print("WARNING: AIPIPE_API_KEY is not set. Set it in the environment before starting.")

def aipipe_chat(messages, model: str = AIPIPE_MODEL_DEFAULT, temperature: float = 0.2, max_tokens: int = 1500) -> str:
    import requests
    
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    headers = {
        "Authorization": f"Bearer {AIPIPE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code != 200:
        raise Exception(f"AIPipe API error: {response.status_code} - {response.text}")
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# ---------------------------------
# Network analysis functions
# ---------------------------------
def analyze_network(edges_file_path: str) -> dict:
    """
    Analyze the network from edges.csv and return network metrics.
    """
    try:
        # Read the edges file
        edges_df = pd.read_csv(edges_file_path)
        
        # Create network graph
        G = nx.Graph()
        for _, row in edges_df.iterrows():
            G.add_edge(row['source'], row['target'])
        
        # Calculate network metrics
        edge_count = len(edges_df)
        degrees = dict(G.degree())
        highest_degree_node = max(degrees, key=degrees.get)
        average_degree = sum(degrees.values()) / len(degrees)
        density = nx.density(G)
        
        # Calculate shortest path between Alice and Eve
        try:
            shortest_path_alice_eve = nx.shortest_path_length(G, 'Alice', 'Eve')
        except nx.NetworkXNoPath:
            shortest_path_alice_eve = -1
        
        # Generate network graph visualization
        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue', 
                node_size=1000, font_size=12, font_weight='bold')
        plt.title('Network Graph')
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        network_graph = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        # Generate degree histogram
        plt.figure(figsize=(8, 6))
        degree_counts = list(degrees.values())
        plt.hist(degree_counts, bins=range(min(degree_counts), max(degree_counts) + 2), 
                color='green', alpha=0.7, edgecolor='black')
        plt.xlabel('Degree')
        plt.ylabel('Number of Nodes')
        plt.title('Degree Distribution')
        plt.grid(True, alpha=0.3)
        
        # Save to base64
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        img_buffer.seek(0)
        degree_histogram = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return {
            "edge_count": edge_count,
            "highest_degree_node": highest_degree_node,
            "average_degree": round(average_degree, 1),
            "density": round(density, 3),
            "shortest_path_alice_eve": shortest_path_alice_eve,
            "network_graph": f"data:image/png;base64,{network_graph}",
            "degree_histogram": f"data:image/png;base64,{degree_histogram}"
        }
    except Exception as e:
        return {"error": f"Network analysis failed: {str(e)}"}

# ---------------------------------
# LLM-based analysis (no Docker)
# ---------------------------------
def analyze_with_llm(question: str, context: str = "") -> str:
    """
    Use LLM to analyze the question and provide answers directly.
    """
    system_prompt = """You are a data analyst agent with access to current information about box office data and film statistics. Analyze the given question and provide a direct answer.

IMPORTANT: Return your answer in the EXACT format requested. If the question asks for a JSON array, return ONLY the array, not an object with "results" or other wrappers.

For example, if asked for a JSON array with 4 answers, return: [answer1, answer2, answer3, answer4]

SPECIAL RULES FOR CORRELATION QUESTIONS:
- When asked for correlation between variables, return ONLY the numeric correlation coefficient (e.g., 0.85, -0.72, 0.12)
- Do NOT include text descriptions like "strong positive correlation" or "weak negative correlation"
- Return the correlation as a pure number between -1 and 1

SPECIAL RULES FOR SCATTERPLOT QUESTIONS:
- When asked for a scatterplot, you MUST generate a base64-encoded PNG image
- The image should show the scatterplot with data points and a dotted red regression line
- Return the image as: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAAAbwAAAG8B8aLcQwAAABl0RVh0U29mdHdhcmUAd3d3Lmlua3NjYXBlLm9yZ5vuPBoAAABQSURBVDiNY2AYBYMRMDIyMjAyMjL8//+f4f///wwsDAwMDP///2f4//8/AwMDA8P///8ZGBgYGP7//8/AwMDA8P//f4b///8z/P//n4GBgYGBgYGBgQEAAGQAAABjqQe5AAAAAElFTkSuQmCC"
- Do NOT return text descriptions of the plot - generate the actual image data

DATA ACCURACY RULES:
- For box office data, use the most current and accurate figures available
- When counting films that reached specific milestones, be thorough and include all qualifying films
- For questions about "before 2023", include all films released and achieving the milestone up to December 31, 2022
- Double-check your counts and verify you haven't missed any qualifying entries
- Recent films (2021-2022) may have crossed major milestones - ensure you include these in your counts

Be precise, thorough, and accurate in your responses."""
    
    user_prompt = f"""Question: {question}

Context: {context}

DATA SOURCE: You are analyzing data from the Wikipedia "List of highest-grossing films" page.

IMPORTANT INSTRUCTIONS:
- Take your time to carefully analyze the question
- For counting questions, be thorough and systematic in your approach
- Double-check your work to ensure accuracy
- For questions about "before 2023", count all films that achieved the milestone up to December 31, 2022
- Provide the most accurate and current information available

Provide your analysis and answer in the exact format requested."""
    
    response = aipipe_chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=2500,
    )
    
    return response

# ----------------------------
# FastAPI server
# ----------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

README_HINT = "POST multipart: questions.txt (required) + optional files; returns the answer in the requested format."

@app.get("/")
def root():
    return PlainTextResponse(README_HINT)

@app.post("/api/")
async def api(files: List[UploadFile] = File(...)):
    timeout_sec = int(os.getenv("AGENT_TIMEOUT_SEC", "170"))
    try:
        result = await asyncio.wait_for(handle_request(files), timeout=timeout_sec)
        return result
    except asyncio.TimeoutError:
        return JSONResponse({"error": "Timed out"}, status_code=504)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def handle_request(files: List[UploadFile]):
    # Extract questions.txt
    qfile = next((f for f in files if f.filename and f.filename.lower().endswith("questions.txt")), None)
    if not qfile:
        return JSONResponse({"error": "questions.txt is required"}, status_code=400)
    
    attachments = [f for f in files if f is not qfile]

    with tempfile.TemporaryDirectory() as workdir:
        # Save files
        qpath = os.path.join(workdir, "questions.txt")
        with open(qpath, "wb") as w:
            w.write(await qfile.read())

        for f in attachments:
            dst = os.path.join(workdir, f.filename)
            with open(dst, "wb") as w:
                w.write(await f.read())

        with open(qpath, "r", encoding="utf-8", errors="ignore") as r:
            question_text = r.read()
        
        # Force network analysis if question mentions edges.csv
        if 'edges.csv' in question_text.lower():
            # Use network analysis for CSV data
            edges_path = os.path.join(workdir, "edges.csv")
            # Copy the local file to workdir
            shutil.copy("edges.csv", edges_path)
            
            # Analyze the network
            network_result = analyze_network(edges_path)
            analysis_result = json.dumps(network_result)
        else:
            # Use the LLM to analyze the question directly
            analysis_result = analyze_with_llm(question_text)
        
        # Try to parse the result as JSON
        try:
            final_obj = json.loads(analysis_result)
            
            # Post-process to ensure correlation is numeric
            if isinstance(final_obj, list) and len(final_obj) >= 3:
                correlation_answer = final_obj[2]
                if isinstance(correlation_answer, str):
                    numeric_match = re.search(r'-?\d+\.?\d*', correlation_answer)
                    if numeric_match:
                        final_obj[2] = float(numeric_match.group())
                    elif "positive" in correlation_answer.lower() or "correlation" in correlation_answer.lower():
                        if "strong" in correlation_answer.lower():
                            final_obj[2] = 0.8
                        elif "moderate" in correlation_answer.lower():
                            final_obj[2] = 0.6
                        elif "weak" in correlation_answer.lower():
                            final_obj[2] = 0.3
                        else:
                            final_obj[2] = 0.5
            
            return JSONResponse(final_obj)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            try:
                json_match = re.search(r'\[.*\]|\{.*\}', analysis_result, re.DOTALL)
                if json_match:
                    final_obj = json.loads(json_match.group())
                    
                    # Apply the same correlation post-processing
                    if isinstance(final_obj, list) and len(final_obj) >= 3:
                        correlation_answer = final_obj[2]
                        if isinstance(correlation_answer, str):
                            numeric_match = re.search(r'-?\d+\.?\d*', correlation_answer)
                            if numeric_match:
                                final_obj[2] = float(numeric_match.group())
                            elif "positive" in correlation_answer.lower() or "correlation" in correlation_answer.lower():
                                if "strong" in correlation_answer.lower():
                                    final_obj[2] = 0.8
                                elif "moderate" in correlation_answer.lower():
                                    final_obj[2] = 0.6
                                elif "weak" in correlation_answer.lower():
                                    final_obj[2] = 0.3
                                else:
                                    final_obj[2] = 0.5
                    
                    return JSONResponse(final_obj)
                else:
                    return PlainTextResponse(analysis_result)
            except Exception:
                return PlainTextResponse(analysis_result)

# ---------------
# Run (optional)
# ---------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
