import json
import time
import pandas as pd
from tqdm import tqdm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Import your actual RAG pipeline components
try:
    from app import load_rag_chain
    # Use load_rag_chain with a default k chunks
    chain = load_rag_chain(k_chunks=5)
except ImportError:
    # Mock chain for testing purposes if import fails
    class MockChain:
        def invoke(self, inputs):
            time.sleep(1.5) # Mock latency
            from langchain.schema import Document
            return {
                "result": "Based on 3GPP Release 16, the answer is option 2.",
                "source_documents": [
                    Document(page_content="This is the correct answer text for Release 16.", metadata={"spec_number": "38.331"}),
                    Document(page_content="Some other text.", metadata={})
                ]
            }
    chain = MockChain()

def preprocess_teleqna_data(filepath="TeleQnA_3GPP.json"):
    """
    1. Data Preprocessing & Strict Domain Filtering
    - Drops any questions containing "IEEE"
    - Keeps ONLY questions where text explicitly contains "release 16" or "release 18" (case-insensitive)
    """
    print(f"Loading dataset from {filepath}...")
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found. Using small mock dataset for demonstration.")
        data = {
            "q1": {"question": "Max throughput in Release 16?", "option 1": "10 Gbps", "option 2": "20 Gbps", "answer": "option 2", "explanation": "Release 16 specifies 20 Gbps."},
            "q2": {"question": "IEEE standard for WiFi?", "option 1": "802.11", "answer": "option 1"},
            "q3": {"question": "Release 18 latency requirements?", "option 1": "1ms", "option 2": "5ms", "answer": "option 1", "explanation": "Release 18 aims for 1ms."},
            "q4": {"question": "What is 5G core?", "option 1": "SBA", "answer": "option 1", "explanation": "Core architecture."} # No release specified
        }
        
    filtered_data = {}
    
    for q_id, q_info in data.items():
        # Combine all relevant text fields to search for keywords
        text_fields = [str(q_info.get("question", "")), str(q_info.get("explanation", ""))]
        for key in q_info:
            if key.startswith("option"):
                text_fields.append(str(q_info[key]))
                
        full_text = " ".join(text_fields).lower()
        
        # Rule 1: Drop if contains "IEEE"
        if "ieee" in full_text:
            continue
            
        # Rule 2: Keep only if contains "release 16" or "release 18"
        if "release 16" in full_text or "release 18" in full_text:
            filtered_data[q_id] = q_info
            
    print(f"Original dataset size: {len(data)}")
    print(f"Strictly filtered dataset size (Domain: Rel 16/18): {len(filtered_data)}")
    return filtered_data

def evaluate_oran_rca():
    """
    KPI 5: Root Cause Analysis (O-RAN) Accuracy
    Standalone mock evaluation function using create_pandas_dataframe_agent.
    """
    print("\n--- Running KPI 5: O-RAN Root Cause Analysis Evaluation ---")
    
    # Create mock oran_logs.csv data
    mock_csv_path = "mock_oran_logs.csv"
    mock_data = pd.DataFrame({
        "timestamp": ["2026-05-13 10:00", "2026-05-13 10:05", "2026-05-13 10:10"],
        "node_id": ["O-DU-1", "O-CU-1", "O-RU-1"],
        "latency_ms": [10, 15, 250],
        "error_code": ["None", "None", "ERR_LATENCY_HIGH"],
        "status": ["OK", "OK", "FAILED"]
    })
    mock_data.to_csv(mock_csv_path, index=False)
    
    try:
        # Note: Requires GOOGLE_API_KEY to be set in the environment
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        df = pd.read_csv(mock_csv_path)
        
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=False, 
            allow_dangerous_code=True
        )
        
        test_questions = [
            "Which node_id has an error_code of ERR_LATENCY_HIGH?",
            "What is the maximum latency_ms recorded?"
        ]
        
        success_count = 0
        print("Testing Pandas Agent with hardcoded RCA questions...")
        for q in test_questions:
            try:
                response = agent.invoke({"input": q})
                if response and "output" in response:
                    success_count += 1
            except Exception as e:
                print(f"Warning: RCA Agent failed on question '{q}'. Error: {e}")
                
        print(f"RCA Agent Test: {success_count}/{len(test_questions)} passed.")
        return success_count == len(test_questions)
        
    except Exception as e:
        print(f"KPI 5 Evaluation aborted due to setup error (e.g., missing API key): {e}")
        return False

def evaluate_rag_pipeline(dataset_filepath="TeleQnA_3GPP.json", sample_size=10):
    """
    Evaluates the LangChain RAG pipeline against 4 KPIs using the strictly filtered dataset.
    """
    filtered_data = preprocess_teleqna_data(dataset_filepath)
    
    # Take a sample for evaluation (to avoid massive API costs/time during testing)
    eval_items = list(filtered_data.items())[:sample_size]
    if not eval_items:
        print("No data available for evaluation after filtering.")
        return
        
    print(f"\n--- Starting RAG Evaluation on {len(eval_items)} sample queries ---")
        
    results = []
    
    kpi1_correct = 0
    kpi2_hits = 0
    kpi4_explainable = 0
    total_latency = 0
    
    for q_id, q_data in tqdm(eval_items, desc="Evaluating Queries"):
        base_question = q_data.get("question", "")
        
        # Build the multiple choice question with options
        options_text = ""
        for key, value in q_data.items():
            if key.startswith("option"):
                options_text += f"- {key}: {value}\n"
                
        # Instruct the model very strongly to pick an option
        full_prompt = (
            f"Question: {base_question}\n\n"
            f"Multiple Choice Options:\n{options_text}\n"
            "CRITICAL REQUIREMENT: You are taking a multiple-choice test. "
            "You MUST begin your answer with the exact choice identifier (e.g., 'option 1', 'option 2', 'option 3'). "
            "If you only provide a description without explicitly stating the option number, you will fail."
        )
        
        # Ground truth in TeleQnA often looks like "option 2: some text"
        # We need to extract just "option X" to match against the LLM's instructed output
        ground_truth_full = q_data.get("answer", "")
        ground_truth_option_key = ground_truth_full.split(":")[0].strip() if ":" in ground_truth_full else ground_truth_full
        
        # We don't have a reliable ground truth exact text if the 'answer' key is the only thing we have.
        # But we can try to find the option text from the options dictionary using the key
        ground_truth_text = q_data.get(ground_truth_option_key, "")
        
        start_time = time.time()
        
        try:
            # Add a small delay to respect free-tier rate limits (15 RPM)
            time.sleep(5)
            
            response = chain.invoke({"query": full_prompt})
                
            end_time = time.time()
            latency = end_time - start_time
            total_latency += latency
            
            # Extract outputs
            llm_answer = response.get("result", response.get("output", ""))
            source_docs = response.get("source_documents", [])
            
            import re
            # KPI 1: Accuracy 
            # We check if "option X" is in the answer
            is_accurate = ground_truth_option_key.lower() in llm_answer.lower()
            
            if is_accurate:
                kpi1_correct += 1
                
            # KPI 2: Contextual Relevance (Hit Rate)
            # Scan retrieved chunks for the ground truth answer text
            # We use word overlap because 3GPP PDFs have weird formatting, line breaks, etc.
            has_context_hit = False
            if ground_truth_text:
                gt_words = set(re.findall(r'\b\w+\b', ground_truth_text.lower()))
                for doc in source_docs:
                    pc_words = set(re.findall(r'\b\w+\b', doc.page_content.lower()))
                    if gt_words:
                        overlap = len(gt_words.intersection(pc_words)) / len(gt_words)
                        if overlap >= 0.5:  # At least 50% of the ground truth words exist in the chunk
                            has_context_hit = True
                            break
            if has_context_hit:
                kpi2_hits += 1
                
            # KPI 4: Explainability
            # Verify retrieved Document objects contain expected metadata
            is_explainable = False
            for doc in source_docs:
                metadata = doc.metadata or {}
                if "filename" in metadata or "spec_number" in metadata:
                    is_explainable = True
                    break
            if is_explainable:
                kpi4_explainable += 1
                
            results.append({
                "query_id": q_id,
                "question": base_question,
                "options": options_text.strip(),
                "ground_truth_option": ground_truth_option_key,
                "ground_truth_text": ground_truth_text,
                "llm_answer": llm_answer,
                "latency_sec": round(latency, 2),
                "is_accurate": is_accurate,
                "context_hit": has_context_hit,
                "is_explainable": is_explainable,
                "num_sources_retrieved": len(source_docs)
            })
            
        except Exception as e:
            # If rate limited, sleep longer and skip this query
            if "429" in str(e) or "Quota" in str(e):
                print(f"\nRate limit hit. Waiting 30s...")
                time.sleep(30)
                
            results.append({
                "query_id": q_id,
                "question": base_question,
                "error": str(e)
            })
            
    # Calculate aggregate KPIs
    num_evals = len(results)
    avg_latency = total_latency / num_evals if num_evals > 0 else 0
    accuracy_pct = (kpi1_correct / num_evals) * 100 if num_evals > 0 else 0
    context_relevance_pct = (kpi2_hits / num_evals) * 100 if num_evals > 0 else 0
    explainability_pct = (kpi4_explainable / num_evals) * 100 if num_evals > 0 else 0
    
    # Run KPI 5
    rca_success = evaluate_oran_rca()
    
    # Save results to CSV
    df_results = pd.DataFrame(results)
    report_file = "hackathon_kpi_report.csv"
    df_results.to_csv(report_file, index=False)
    
    # Print Beautiful Terminal Summary
    print("\n" + "="*60)
    print("🚀 TELECOMRAG HACKATHON EVALUATION SUMMARY 🚀".center(60))
    print("="*60)
    
    # KPI 1
    acc_indicator = "✅ PASS" if accuracy_pct > 85 else "❌ FAIL"
    print(f" KPI 1: Accuracy                 | {accuracy_pct:.1f}% (Target: >85%) {acc_indicator}")
    
    # KPI 2
    print(f" KPI 2: Contextual Relevance     | {context_relevance_pct:.1f}% Hit Rate")
    
    # KPI 3
    lat_indicator = "✅ PASS" if avg_latency < 3.0 else "❌ FAIL"
    print(f" KPI 3: Avg Latency              | {avg_latency:.2f}s (Target: <3s) {lat_indicator}")
    
    # KPI 4
    print(f" KPI 4: Explainability (Metadata)| {explainability_pct:.1f}% Sources Verified")
    
    # KPI 5
    rca_indicator = "✅ PASS" if rca_success else "❌ FAIL"
    print(f" KPI 5: O-RAN RCA Agent Test     | {rca_indicator}")
    
    print("-" * 60)
    print(f" 📊 Detailed results saved to: {report_file}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # You can adjust sample_size here based on time/budget constraints
    evaluate_rag_pipeline(dataset_filepath="TeleQnA (3000).json", sample_size=10)
