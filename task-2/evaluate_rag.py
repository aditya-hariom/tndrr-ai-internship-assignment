"""
Evaluation Runner for RAG from Scratch
Tests 15 verified questions across two chunking strategies (Small vs Large).
Computes Recall@5, Precision@5, MRR, and displays retrieved passages.
"""

import os
import json
import time
from tabulate import tabulate
from rag_pipeline import (
    DocumentLoader,
    TextPreprocessor,
    Chunker,
    EmbeddingEngine,
    VectorIndex,
    BM25Retriever,
    LLMGenerator,
    evaluate_retrieval
)

# ============================================================================
# 15 ORIGINAL VERIFIED EVALUATION QUESTIONS
# ============================================================================

EVALUATION_QUESTIONS = [
    {
        "id": 1,
        "type": "Direct Factual",
        "question": "What are the four core functions that organize AI risk management activities in the NIST AI RMF Core?",
        "correct_answer": "The four core functions are GOVERN, MAP, MEASURE, and MANAGE.",
        "exact_source_passage": "the Core is composed of four functions: GOVERN, MAP, MEASURE, and MANAGE. Each of these high-level functions is broken down into categories and subcategories.",
        "target_page": 25,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 2,
        "type": "Direct Factual",
        "question": "According to the NIST AI RMF, what are the three major categories of AI bias that must be considered and managed?",
        "correct_answer": "The three major categories of AI bias are systemic, computational and statistical, and human-cognitive.",
        "exact_source_passage": "NIST has identified three major categories of AI bias to be considered and managed: systemic, computational and statistical, and human-cognitive.",
        "target_page": 23,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 3,
        "type": "Understanding",
        "question": "Why is GOVERN described as a cross-cutting function in the AI RMF Core?",
        "correct_answer": "GOVERN is cross-cutting because it is infused throughout AI risk management, enables and connects the other functions (MAP, MEASURE, MANAGE), and ensures continual risk governance across organizational leadership, culture, and lifecycle processes.",
        "exact_source_passage": "GOVERN is a cross-cutting function that is infused throughout AI risk management and enables the other functions of the process. Aspects of GOVERN, especially those related to compliance or evaluation, are often best integrated into... risk culture.",
        "target_page": 27,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 4,
        "type": "Understanding",
        "question": "How does the NIST AI RMF define the distinction between explainability and interpretability in AI systems?",
        "correct_answer": "Explainability refers to a representation of the mechanisms underlying AI systems' operation, whereas interpretability refers to the meaning of AI systems' output in the context of their designed functional purposes.",
        "exact_source_passage": "Explainability refers to a representation of the mechanisms underlying AI systems’ operation, whereas interpretability refers to the meaning of AI systems’ output in the context of their designed functional purposes.",
        "target_page": 21,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 5,
        "type": "Synthesis of Information",
        "question": "How does the MAP function provide the foundational basis for carrying out the MEASURE and MANAGE functions?",
        "correct_answer": "The MAP function establishes contextual understanding of the AI system, its actors, and potential impacts; without this contextual knowledge, measuring AI risks and managing or prioritizing treatments cannot be effectively performed.",
        "exact_source_passage": "The information gathered while carrying out the MAP function enables negative risk prevention and informs decisions for mapping, measuring, and managing AI risks... Outcomes in the MAP function are the basis for the MEASURE and MANAGE functions. Without contextual knowledge of AI system risks, risk management is difficult to perform.",
        "target_page": 30,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 6,
        "type": "Specific Terminology",
        "question": "How does the AI RMF define 'risk tolerance' and does the Framework itself prescribe specific risk tolerance levels?",
        "correct_answer": "Risk tolerance refers to an organization's or AI actor's readiness to bear risk to achieve its objectives, and the AI RMF does not prescribe risk tolerance because it depends on specific organizational priorities, legal requirements, and context.",
        "exact_source_passage": "While the AI RMF can be used to prioritize risk, it does not prescribe risk tolerance. Risk tolerance refers to the organization’s or AI actor’s (see Appendix A) readiness to bear the risk in order to achieve its objectives.",
        "target_page": 12,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 7,
        "type": "Specific Terminology",
        "question": "What is 'negative residual risk' as defined in category MANAGE 1.4 of the AI RMF?",
        "correct_answer": "Negative residual risk is defined as the sum of all remaining negative risks after risk treatments have been applied.",
        "exact_source_passage": "MANAGE 1.4: Negative residual risks (defined as the sum of all remaining negative risks after treatments have been applied) are evaluated and either tolerated, mitigated, or transferred, or the AI system is phased out.",
        "target_page": 37,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 8,
        "type": "Cross-Sectional / Multi-Section Synthesis",
        "question": "How do the risks associated with AI systems differ from traditional software risks according to Appendix B?",
        "correct_answer": "AI risks differ because AI systems rely heavily on training data that may be non-representative or ungrounded, utilize opaque/black-box models, exhibit statistical non-determinism, face emergent failure modes, and incur massive computational and environmental costs.",
        "exact_source_passage": "Compared to traditional software, AI-specific risks that are new or increased include the following: The data used for building an AI system may not be a true or appropriate representation of the context or intended use... Machine learning models are often opaque or black-box... High degree of statistical variance and non-determinism... Computational costs for developing AI systems and their impact on the environment.",
        "target_page": 43,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 9,
        "type": "Direct Factual",
        "question": "What are the seven key characteristics of trustworthy AI systems identified in Section 3 of the NIST AI RMF?",
        "correct_answer": "The seven characteristics are: 1) Valid and Reliable, 2) Safe, 3) Secure and Resilient, 4) Accountable and Transparent, 5) Explainable and Interpretable, 6) Privacy-Enhanced, and 7) Fair – with Harmful Bias Managed.",
        "exact_source_passage": "Characteristics of trustworthy AI systems include: valid and reliable, safe, secure and resilient, accountable and transparent, explainable and interpretable, privacy-enhanced, and fair with harmful bias managed.",
        "target_page": 17,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 10,
        "type": "Understanding",
        "question": "What are the three potential levels or categories of harm resulting from AI systems illustrated in Figure 1?",
        "correct_answer": "The three potential levels of harm are harm to people (individuals and groups), harm to an organization, and harm to ecosystems (society, environment, and global infrastructure).",
        "exact_source_passage": "Fig. 1. Examples of potential harms related to AI systems. Trustworthy AI systems and their responsible use can mitigate negative risks and contribute to benefits for people, organizations, and ecosystems.",
        "target_page": 10,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 11,
        "type": "Understanding",
        "question": "What is the primary role and methodological approach of the MEASURE function within the AI RMF Core?",
        "correct_answer": "The MEASURE function employs quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts.",
        "exact_source_passage": "The MEASURE function employs quantitative, qualitative, or mixed-method tools, techniques, and methodologies to analyze, assess, benchmark, and monitor AI risk and related impacts. It uses knowledge relevant to AI risks identified in the MAP function and informs the MANAGE function.",
        "target_page": 33,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 12,
        "type": "Specific Terminology",
        "question": "What are AI RMF Profiles as described in Section 6, and what is the difference between use-case and temporal profiles?",
        "correct_answer": "AI RMF Profiles are implementations of Core functions, categories, and subcategories for specific applications or settings; use-case profiles address specific sectors (e.g. hiring, fair housing), while temporal profiles compare current state vs desired target state to reveal gaps.",
        "exact_source_passage": "AI RMF use-case profiles are implementations of the AI RMF functions, categories, and subcategories for a specific setting or application based on the requirements, risk tolerance, and resources of the Framework user... AI RMF temporal profiles are descriptions of either the current state or the desired, target state.",
        "target_page": 38,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 13,
        "type": "Synthesis of Information",
        "question": "How are third-party resources and pre-trained models managed under the MANAGE function (MANAGE 3)?",
        "correct_answer": "Under MANAGE 3, AI risks and benefits from third-party resources are identified and managed, and pre-trained models used for development are assessed for inherited risks and vulnerabilities.",
        "exact_source_passage": "MANAGE 3: AI risks and benefits from third-party resources are identified and managed. MANAGE 3.1: AI risks and benefits from third-party resources are regularly monitored and managed. MANAGE 3.2: Pre-trained models which are used for development are assessed for risk.",
        "target_page": 37,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 14,
        "type": "Direct Factual / Nuanced Understanding",
        "question": "Does the NIST AI RMF state that bias is always a negative phenomenon?",
        "correct_answer": "No, the AI RMF states that bias is not always a negative phenomenon; however, AI systems can increase the speed and scale of biases and amplify harms to individuals, communities, and society.",
        "exact_source_passage": "While bias is not always a negative phenomenon, AI systems can potentially increase the speed and scale of biases and perpetuate and amplify harms to individuals, groups, communities, organizations, and society.",
        "target_page": 23,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    },
    {
        "id": 15,
        "type": "Specific Terminology / Attributes",
        "question": "What key development attributes of the AI RMF are outlined in Appendix D?",
        "correct_answer": "The Framework strives to be risk-based, voluntary, consensus-driven, transparent, clear and plain in language, easily usable, and a living document that is updated over time.",
        "exact_source_passage": "The AI RMF strives to: 1. Be risk-based, resource-efficient, pro-innovation, and voluntary. 2. Be consensus-driven and developed and regularly updated through an open, transparent process... living document.",
        "target_page": 47,
        "doc_name": "NIST_AI_RMF_1.0.pdf"
    }
]


def run_experiment():
    print("=" * 80)
    print("STARTING EXPERIMENT: RAG FROM SCRATCH ON NIST AI RMF 1.0")
    print("=" * 80)

    # 1. Document Loading
    pdf_path = "NIST_AI_RMF_1.0.pdf"
    loader = DocumentLoader(pdf_path)
    raw_pages = loader.load()
    print(f"Loaded {len(raw_pages)} pages from {pdf_path}.")

    # 2. Text Preprocessing
    cleaned_pages = []
    for p in raw_pages:
        cleaned_text = TextPreprocessor.clean_text(p["text"])
        cleaned_pages.append({
            **p,
            "cleaned_text": cleaned_text
        })
    total_cleaned_words = sum(len(p["cleaned_text"].split()) for p in cleaned_pages)
    print(f"Preprocessed {len(cleaned_pages)} pages. Total word count: {total_cleaned_words:,} words.")

    # 3. Chunking Configurations
    # Config 1: Small Chunks (~250 words, 50-word overlap)
    small_chunker = Chunker(chunk_size_words=250, overlap_words=50, config_name="small")
    small_chunks = small_chunker.chunk_document(cleaned_pages)

    # Config 2: Large Chunks (~500 words, 100-word overlap)
    large_chunker = Chunker(chunk_size_words=500, overlap_words=100, config_name="large")
    large_chunks = large_chunker.chunk_document(cleaned_pages)

    print(f"Generated {len(small_chunks)} Small Chunks (size=250, overlap=50).")
    print(f"Generated {len(large_chunks)} Large Chunks (size=500, overlap=100).")

    # 4. Embeddings Generation
    embedder = EmbeddingEngine(model_name="all-MiniLM-L6-v2")

    print("\nEncoding Small Chunks...")
    small_texts = [c["text"] for c in small_chunks]
    small_embeddings = embedder.embed_texts(small_texts, normalize=True)

    print("Encoding Large Chunks...")
    large_texts = [c["text"] for c in large_chunks]
    large_embeddings = embedder.embed_texts(large_texts, normalize=True)

    # 5. Build Vector Indexes
    small_index = VectorIndex(small_embeddings, small_chunks, use_faiss=True)
    large_index = VectorIndex(large_embeddings, large_chunks, use_faiss=True)

    # Also build BM25 for comparative retrieval analysis
    bm25_small = BM25Retriever(small_chunks)

    # Initialize LLM Generator
    generator = LLMGenerator()

    # 6. Run Evaluation on 15 Questions
    results = {
        "metadata": {
            "dataset": "NIST AI 100-1 (AI RMF 1.0)",
            "total_pages": len(raw_pages),
            "small_chunk_count": len(small_chunks),
            "large_chunk_count": len(large_chunks),
            "embedding_model": "all-MiniLM-L6-v2",
            "eval_date": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "questions": []
    }

    small_hits = 0
    large_hits = 0
    small_rr_sum = 0.0
    large_rr_sum = 0.0
    small_p5_sum = 0.0
    large_p5_sum = 0.0

    comparison_rows = []

    print("\n" + "=" * 80)
    print("RUNNING RETRIEVAL EVALUATION OVER 15 ORIGINAL QUESTIONS")
    print("=" * 80)

    for q in EVALUATION_QUESTIONS:
        qid = q["id"]
        qtext = q["question"]
        tpage = q["target_page"]
        gt_passage = q["exact_source_passage"]

        # Embed Query
        q_vec = embedder.embed_query(qtext, normalize=True)

        # Retrieve Top-5 for Small Chunks
        small_retrieved = small_index.retrieve(q_vec, top_k=5)
        small_eval = evaluate_retrieval(small_retrieved, tpage, gt_passage, top_k=5)

        # Retrieve Top-5 for Large Chunks
        large_retrieved = large_index.retrieve(q_vec, top_k=5)
        large_eval = evaluate_retrieval(large_retrieved, tpage, gt_passage, top_k=5)

        # BM25 Retrieval for comparison
        bm25_retrieved = bm25_small.retrieve(qtext, top_k=5)
        bm25_eval = evaluate_retrieval(bm25_retrieved, tpage, gt_passage, top_k=5)

        # Generate sample LLM answer using top context from best retriever
        sample_retrieved = small_retrieved if small_eval["is_hit"] else large_retrieved
        llm_response = generator.generate_answer(qtext, sample_retrieved)

        # Accumulate metrics
        if small_eval["is_hit"]:
            small_hits += 1
        if large_eval["is_hit"]:
            large_hits += 1

        small_rr_sum += small_eval["reciprocal_rank"]
        large_rr_sum += large_eval["reciprocal_rank"]
        small_p5_sum += small_eval["precision_at_k"]
        large_p5_sum += large_eval["precision_at_k"]

        # Determine note for comparison table
        s_hit_str = f"Yes (Rank {small_eval['hit_rank']})" if small_eval['is_hit'] else "No"
        l_hit_str = f"Yes (Rank {large_eval['hit_rank']})" if large_eval['is_hit'] else "No"

        notes = []
        if small_eval['is_hit'] and large_eval['is_hit']:
            if small_eval['hit_rank'] < large_eval['hit_rank']:
                notes.append("Small chunk ranked higher")
            elif large_eval['hit_rank'] < small_eval['hit_rank']:
                notes.append("Large chunk ranked higher")
            else:
                notes.append("Both retrieved at Rank 1")
        elif small_eval['is_hit'] and not large_eval['is_hit']:
            notes.append("Only Small chunk retrieved")
        elif large_eval['is_hit'] and not small_eval['is_hit']:
            notes.append("Only Large chunk retrieved (broader context)")
        else:
            notes.append("Both failed (hard query)")

        comparison_rows.append([
            f"Q{qid:02d}: {qtext[:45]}...",
            s_hit_str,
            l_hit_str,
            "; ".join(notes)
        ])

        # Record question details
        q_record = {
            "id": qid,
            "type": q["type"],
            "question": qtext,
            "target_page": tpage,
            "correct_answer": q["correct_answer"],
            "exact_source_passage": gt_passage,
            "small_chunk_results": {
                "is_hit": small_eval["is_hit"],
                "hit_rank": small_eval["hit_rank"],
                "mrr": small_eval["reciprocal_rank"],
                "precision_at_5": small_eval["precision_at_k"],
                "top_retrieved": [
                    {
                        "chunk_id": c["chunk_id"],
                        "score": round(score, 4),
                        "page": c["primary_page"],
                        "pages": c["page_numbers"],
                        "text_snippet": c["text"][:180] + "..."
                    }
                    for c, score in small_retrieved
                ]
            },
            "large_chunk_results": {
                "is_hit": large_eval["is_hit"],
                "hit_rank": large_eval["hit_rank"],
                "mrr": large_eval["reciprocal_rank"],
                "precision_at_5": large_eval["precision_at_k"],
                "top_retrieved": [
                    {
                        "chunk_id": c["chunk_id"],
                        "score": round(score, 4),
                        "page": c["primary_page"],
                        "pages": c["page_numbers"],
                        "text_snippet": c["text"][:180] + "..."
                    }
                    for c, score in large_retrieved
                ]
            },
            "bm25_small_results": {
                "is_hit": bm25_eval["is_hit"],
                "hit_rank": bm25_eval["hit_rank"]
            },
            "generated_answer": llm_response["answer"],
            "llm_provider": llm_response["provider"]
        }
        results["questions"].append(q_record)

        print(f"[{qid:02d}/15] {qtext[:60]}... -> Small: {s_hit_str} | Large: {l_hit_str}")

    # Summary Metrics
    num_q = len(EVALUATION_QUESTIONS)
    small_recall = small_hits / num_q
    large_recall = large_hits / num_q
    small_mrr = small_rr_sum / num_q
    large_mrr = large_rr_sum / num_q
    small_p5 = small_p5_sum / num_q
    large_p5 = large_p5_sum / num_q

    results["summary"] = {
        "num_questions": num_q,
        "small_chunks": {
            "chunk_size_words": 250,
            "overlap_words": 50,
            "total_chunks": len(small_chunks),
            "recall_at_5": round(small_recall, 4),
            "mrr": round(small_mrr, 4),
            "precision_at_5": round(small_p5, 4)
        },
        "large_chunks": {
            "chunk_size_words": 500,
            "overlap_words": 100,
            "total_chunks": len(large_chunks),
            "recall_at_5": round(large_recall, 4),
            "mrr": round(large_mrr, 4),
            "precision_at_5": round(large_p5, 4)
        }
    }

    # Save to JSON
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("INDIVIDUAL QUESTION RESULTS TABLE:")
    print("=" * 80)
    headers = ["Question", "Small Chunk: Relevant in Top 5?", "Large Chunk: Relevant in Top 5?", "Notes"]
    print(tabulate(comparison_rows, headers=headers, tablefmt="github"))

    print("\n" + "=" * 80)
    print("OVERALL PERFORMANCE SUMMARY TABLE:")
    print("=" * 80)
    summary_headers = ["Chunk Size", "Recall@5", "Precision@5", "MRR", "Overall Performance Analysis"]
    summary_rows = [
        [
            f"Small chunks (~250 words, {len(small_chunks)} chunks)",
            f"{small_recall:.2%}",
            f"{small_p5:.2%}",
            f"{small_mrr:.4f}",
            "High specificity and pinpoint semantic precision; lower surrounding context"
        ],
        [
            f"Large chunks (~500 words, {len(large_chunks)} chunks)",
            f"{large_recall:.2%}",
            f"{large_p5:.2%}",
            f"{large_mrr:.4f}",
            "Superior context preservation and broader cross-sentence coverage"
        ]
    ]
    print(tabulate(summary_rows, headers=summary_headers, tablefmt="github"))
    print("\n[Complete] Results successfully saved to evaluation_results.json")

    return results

if __name__ == "__main__":
    run_experiment()

