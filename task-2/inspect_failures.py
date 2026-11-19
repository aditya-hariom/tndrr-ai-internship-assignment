import json

with open('evaluation_results.json', 'r', encoding='utf-8') as f:
    res = json.load(f)

for q in res['questions']:
    s_hit = q['small_chunk_results']['is_hit']
    l_hit = q['large_chunk_results']['is_hit']
    if not s_hit or not l_hit:
        print(f"\n=======================================================")
        print(f"QUESTION {q['id']}: {q['question']}")
        print(f"Target Page: {q['target_page']}")
        print(f"Small hit: {s_hit} (Rank: {q['small_chunk_results']['hit_rank']})")
        print(f"Large hit: {l_hit} (Rank: {q['large_chunk_results']['hit_rank']})")
        print(f"Exact ground truth: {q['exact_source_passage']}")
        print("\n--- Small Chunks Retrieved Top 3: ---")
        for i, c in enumerate(q['small_chunk_results']['top_retrieved'][:3], 1):
            print(f"  [{i}] ID: {c['chunk_id']} | Page: {c['page']} | Score: {c['score']:.4f}")
            print(f"      Snippet: {c['text_snippet'][:140]}")
        print("\n--- Large Chunks Retrieved Top 3: ---")
        for i, c in enumerate(q['large_chunk_results']['top_retrieved'][:3], 1):
            print(f"  [{i}] ID: {c['chunk_id']} | Page: {c['page']} | Score: {c['score']:.4f}")
            print(f"      Snippet: {c['text_snippet'][:140]}")

