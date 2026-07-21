
# Question encoder & answer search

#=============================================================================
# First Index loading
# --- LOAD (in any future session) ---
# from final_project_extractionPart import retrieve
from sentence_transformers import SentenceTransformer
import faiss, pickle

def retrieve(question, k=5):
    q_vector = question_encoder.encode([question], convert_to_numpy=True)
    scores, indices = index.search(q_vector, k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "title": passage_titles[idx],
            "passage": passages[idx],
            "score": float(score)
        })
    return results

#=============================================================================

save_dir = "FinalProject/data"
question_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index(f"{save_dir}/passage_index.faiss")
with open(f"{save_dir}/passages_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)
    passages = metadata["passages"]
    passage_titles = metadata["passage_titles"]

print(f"Loaded index with {index.ntotal} vectors and {len(passages)} passages")
#=============================================================================
from datasets import load_dataset
squad = load_dataset("rajpurkar/squad")

# Test it on a real SQuAD question
test_question = squad["validation"][0]["question"]
test_answer = squad["validation"][0]["answers"]["text"][0]

print(f"Question: {test_question}")
print(f"Ground-truth answer: {test_answer}\n")

results = retrieve(test_question)
for r in results:
    print(f"[{r['score']:.3f}] {r['title']}: {r['passage'][:150]}...")

print("\n---\n")



#=============================================================================
#=============================================================================
# Evaluation function for retrieval performance
def evaluate_retrieval(dataset, k=5, encoder=None, index=None, passages=None, sample_size=None):
    """
    dataset: squad['validation'] (or a subset)
    k: number of passages to retrieve per question
    """
    questions = dataset["question"]
    answers = dataset["answers"]

    if sample_size:
        questions = questions[:sample_size]
        answers = answers[:sample_size]

    correct = 0
    total = 0
    results_log = []  # keep for inspection later

    for question, answer_dict in zip(questions, answers):
        gold_answers = answer_dict["text"]  # list of acceptable answer strings
        if not gold_answers:
            continue  # skip unanswerable/empty cases

        # Encode + search
        q_vector = encoder.encode([question], convert_to_numpy=True)
        scores, indices = index.search(q_vector, k)
        retrieved_passages = [passages[idx] for idx in indices[0]]

        # Check if any gold answer appears in any retrieved passage
        found = any(
            gold.lower() in passage.lower()
            for gold in gold_answers
            for passage in retrieved_passages
        )

        correct += int(found)
        total += 1

        results_log.append({
            "question": question,
            "gold_answers": gold_answers,
            "found": found,
            "retrieved_titles": [passage_titles[idx] for idx in indices[0]]
        })

    accuracy = correct / total if total > 0 else 0
    print(f"Retrieval Accuracy@{k}: {accuracy:.4f} ({correct}/{total})")
    return accuracy, results_log
#=============================================================================
# first test run on a small sample of the validation set
print("first small test: \n")

accuracy, logs = evaluate_retrieval(
    dataset=squad["validation"],
    k=5,
    encoder=question_encoder,
    index=index,
    passages=passages,
    sample_size=100  # just first 100 questions for a quick check
)

# Look at a few failures to understand what's going wrong
failures = [r for r in logs if not r["found"]][:5]

for f in failures:
    print(f"Q: {f['question']}")
    print(f"Gold answers: {f['gold_answers']}")
    print(f"Retrieved from: {f['retrieved_titles']}")
    print("---")

#=============================================================================
# real evaluation on the full validation set
print("\n\n---\n\nNow evaluating on the full validation set:\n")
for k in [1, 5, 10]:
    evaluate_retrieval(
        dataset=squad["validation"],
        k=k,
        encoder=question_encoder,
        index=index,
        passages=passages,
        sample_size=500  # keep it moderate for multiple runs
    )
