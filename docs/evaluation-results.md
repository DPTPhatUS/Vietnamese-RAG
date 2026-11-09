# Evaluation results and failure analysis

[Back to README](../README.md#saved-evaluation-results)

The saved evaluations cover the same 400 questions and reference answers from [the benchmark](../data/benchmark/test.json) in four retrieval modes. Hybrid has the highest available-score means for context recall and faithfulness; RAPTOR has the highest answer relevancy. KG's main observed limitation is retrieval coverage: it returned no context for 306 questions. The examples below also show generation failures even when the necessary evidence was retrieved.

This report analyzes existing artifacts; no new model evaluation was run. Example diagnoses come from comparing the saved question, reference, contexts, and answer, rather than treating a low automated score as proof of a particular error.

## Results

Each cell is **mean score (valid scores / 400)**, rounded to four decimal places. Higher is better, and the highest mean in each column is bolded. These are Ragas scores, not answer accuracy percentages.

| Retrieval mode | Answer relevancy | Context recall | Faithfulness |
| --- | --- | --- | --- |
| Hybrid | 0.6501 (399 / 400) | **0.6492** (387 / 400) | **0.8113** (346 / 400) |
| KG | 0.1236 (391 / 400) | 0.2118 (92 / 400) | 0.7122 (79 / 400) |
| RAPTOR | **0.7432** (399 / 400) | 0.2655 (377 / 400) | 0.6871 (357 / 400) |
| Routed | 0.6675 (397 / 400) | 0.3480 (340 / 400) | 0.6939 (313 / 400) |

Sources: [mean scores](../artifacts/evaluate/summary/mean_scores.csv), [valid counts and distributions](../artifacts/evaluate/summary/metric_summary.csv), and [per-question CSVs](../artifacts/evaluate/).

- **Context recall** assesses how much of the reference answer the retrieved contexts support. It is evidence about retrieval coverage of the answer, rather than a direct check of the generated answer.
- **Faithfulness** assesses how much of the generated answer the retrieved contexts support. A faithful answer can still omit the requested information or abstain.
- **Answer relevancy** assesses whether the answer addresses the question. It does not establish factual correctness or source support.

### Retrieval coverage and missing scores

Context-return counts below were recomputed from the [saved QA logs](../artifacts/test/). Returning a context does not mean it contains the answer.

| Retrieval mode | Questions with context | Questions with no context | Context-return rate |
| --- | --- | --- | --- |
| Hybrid | 400 | 0 | 100.00% |
| KG | 94 | 306 | 23.50% |
| RAPTOR | 400 | 0 | 100.00% |
| Routed | 363 | 37 | 90.75% |

All 306 empty-context KG outputs and all 37 empty-context Routed outputs use the same fallback answer: “Xin lỗi, tôi không tìm được thông tin phù hợp trong cơ sở tri thức hiện có.” The [QA pipeline](../vietrag/pipelines/qa.py) returns this text directly when no documents are available. These cases are retrieval failures to supply evidence, followed by an appropriate fallback, rather than evidence of generation hallucination. The saved logs cannot distinguish missing graph knowledge from entity lookup, query, or service failures.

Missing metric values are excluded from the reported means. The [evaluator](../vietrag/pipelines/eval.py) skips samples without a metric's required fields and writes missing values for invalid evaluation outputs. For KG, the 306 empty-context samples have no context recall or faithfulness score; among the remaining 94 samples, another 2 context recall and 15 faithfulness scores are missing. The artifacts do not record the reasons for these remaining gaps.

Consequently, KG's 0.7122 faithfulness describes only 79 scored answers. It does not summarize all 400 questions. The other modes also have missing scores despite nonempty contexts.

As a sensitivity check, restricting faithfulness to the **61 questions with valid scores in all four modes** gives Hybrid 0.7269, KG 0.7133, RAPTOR 0.6896, and Routed 0.7533. This changes which mode has the highest mean. That small, selected subset is not a replacement benchmark; it shows why the available-score means alone cannot establish an overall winner.

## How failures are separated

| Category | Evidence used for diagnosis | What to investigate |
| --- | --- | --- |
| Retrieval error | No context, or the returned passages do not contain the needed evidence even though it exists in the corpus. | Source coverage, entity lookup, search, chunking, and ranking. |
| Generation error | The needed evidence is available in the supplied passages, but the answer contradicts it, omits it, or attributes it to the wrong entity. | Reading the evidence, preserving relations, and grounding answer claims. |
| Mixed failure | Retrieval misses the needed fact and generation then supplies a claim that the returned passages do not support for the requested entity. | Both evidence selection and behavior when evidence is insufficient. |

The [current generator](../vietrag/pipelines/qa.py) limits context text to 8,000 characters, while the logs save the returned passages in full. For the examples below, the combined context text fits within that limit, and the quoted generation evidence appears in the first passage. The artifacts do not save the exact historical prompt, so this check uses the current implementation.

## Representative retrieval errors

The examples are manually selected illustrations of distinct failure patterns, not an exhaustive annotation or an estimate of each pattern's frequency. Sample indices are zero-based and match `sample_index` in the per-question CSVs. Quoted excerpts preserve the saved Vietnamese wording; surrounding explanations summarize the records.

### R1. KG returns no evidence for a definition

**Record:** KG, sample **0** — [saved QA record](../artifacts/test/test_results_kg.json#L3).

- **Question:** “Bệnh ngoại cảm theo Đông y được định nghĩa như thế nào?”
- **Reference:** A definition based on the body's inability to adapt to abnormal environmental and weather conditions.
- **Retrieved evidence:** `contexts: []`.
- **Actual answer:** “Xin lỗi, tôi không tìm được thông tin phù hợp trong cơ sở tri thức hiện có.”
- **Saved scores:** answer relevancy **0.0000**; context recall and faithfulness **missing**.

**Diagnosis:** Retrieval supplied no evidence. This is not an unsupported generated answer: the fallback is consistent with the empty input. The same question is answered with supporting text in [Hybrid's sample 0](../artifacts/test/test_results_hybrid.json#L3), so the information exists in the project's book sources. That does not prove it exists in the graph.

**Lesson:** Measure empty retrieval explicitly and consider a book-retrieval fallback for unanswered KG queries. Investigate graph coverage and lookup behavior before attributing these outputs to the language model.

### R2. RAPTOR retrieves unrelated passages and the answer abstains

**Record:** RAPTOR, sample **2** — [saved QA record](../artifacts/test/test_results_raptor.json#L27).

- **Question:** “Khi nào lục khí được gọi là lục dâm hay lục tà?”
- **Reference:** When the climatic influences become excessive, insufficient, or occur at the wrong time and cause disease.
- **Retrieved evidence:** Five passages about địa chi, Thận, prescriptions, visual diagnosis, and diagnostic history. They do not provide the requested condition.
- **Actual answer excerpt:** “Không có thông tin trong nguồn tham chiếu để xác định khi nào lục khí được gọi là lục dâm hay lục tà.”
- **Saved scores:** context recall **0.0000**, answer relevancy **1.0000**, faithfulness **1.0000**.

**Diagnosis:** Retrieval missed evidence that is present in [book 1](../data/book1/markdown/book1.md#L212). Hybrid's sample 2 retrieves the relevant passage and answers the question. RAPTOR's abstention is reasonable given its retrieved evidence; the failed task should not be counted as a generation hallucination.

**Lesson:** Five retrieved passages do not guarantee useful evidence. Inspect whether the answer-bearing passage survives search and reranking. The perfect relevancy and faithfulness scores on this abstention also show that those metrics alone do not measure successful question answering.

## Representative generation errors

### G1. Hybrid reverses a relation stated in the first passage

**Record:** Hybrid, sample **291** — [saved QA record](../artifacts/test/test_results_hybrid.json#L3495).

- **Question:** “Bệnh của tạng Can thường nặng lên vào mùa nào?”
- **Reference:** “Thường nặng lên vào mùa Thu.”
- **Retrieved evidence, context 1:** “bệnh thuộc tạng can dễ khỏi vào mùa hạ”; later, “thì sang mùa thu sẽ nặng”.
- **Actual answer:** “Bệnh của tạng Can thường nặng lên vào mùa hạ.”
- **Saved scores:** context recall **1.0000**, answer relevancy **0.6765**, faithfulness **0.0000**.

**Diagnosis:** The retrieved passage supplies the expected relation, but the answer uses the season associated with improvement for a question about worsening. The failure occurs in interpreting or expressing the evidence, rather than in retrieving the necessary passage.

**Lesson:** Check the complete subject–relation–value claim. Retrieving text that mentions the correct subject and both possible values is insufficient if generation swaps the relations.

### G2. Hybrid uses a neighboring entity's location

**Record:** Hybrid, sample **187** — [saved QA record](../artifacts/test/test_results_hybrid.json#L2247).

- **Question:** “Huyệt Huyết hải nằm ở đâu?”
- **Reference:** “Nằm ở mặt trong đùi, từ xương bánh chè đo lên 2 thốn.”
- **Retrieved evidence, context 1:** Under the heading “Huyệt hải”, the passage states “Ở cạnh trong đùi, trên đầu gối 2 thốn”. Context 2 separately lists “Thượng Huyết hải” with “Huyệt Huyết hải lên 3 thốn”.
- **Actual answer:** “Huyệt Huyết hải nằm ở cạnh trong đùi, phía trên huyệt Huyệt hải 3 thốn.”
- **Saved scores:** context recall **0.0000**, answer relevancy **0.8713**, faithfulness **0.5000**.

**Diagnosis:** Despite a source spelling variation (“Huyệt hải”), the first passage contains the reference location. The answer instead uses the offset belonging to Thượng Huyết hải. This is primarily an entity attribution error during generation, with noisy source spelling as a contributing concern. The zero context recall score conflicts with the visible supporting passage, so it is not sufficient evidence of a retrieval failure here.

**Lesson:** Preserve entity headings with their facts, distinguish related names, and require evidence for the exact requested entity. Merely copying a location from a retrieved passage does not make its attribution correct.

## A mixed retrieval and generation failure

### M1. Missing target evidence followed by a wrong attribution

**Record:** Hybrid, sample **142** — [saved QA record](../artifacts/test/test_results_hybrid.json#L1707).

- **Question:** “Huyệt Ấn đường nằm ở vị trí nào?”
- **Reference:** “Nằm ở điểm chính giữa đường nối hai đầu lông mày.”
- **Retrieved evidence:** The first passage locates **Tứ Thần thông** with “Ở huyệt Bách hội ra trước, sau, phải, trái cách đều 1 thốn.” It mentions Ấn đường only as a landmark for another point; none of the five passages provides its reference location.
- **Actual answer:** “Huyệt Ấn đường nằm ở vị trí: Ở huyệt Bách hội ra trước, sau, phải, trái cách đều 1 thốn.”
- **Saved scores:** context recall **0.0000**, answer relevancy **0.8558**, faithfulness **0.0000**.

**Diagnosis:** Retrieval misses the target location, which is available in [book 2](../data/book2/markdown/book2.md#L2738). Generation compounds the problem by assigning Tứ Thần thông's location to Ấn đường. These are two separate failures; the lack of target evidence does not justify the unsupported attribution.

**Lesson:** Improve retrieval of the target entity and make the generator abstain when no passage supports that entity's requested attribute. A high relevancy score can coexist with a wrong answer.

## What can be learned

| Observation | Supported lesson | Next experiment |
| --- | --- | --- |
| Hybrid has the highest available-score context recall and faithfulness. | Combining evidence sources is promising in these saved runs, but missing scores affect comparisons. | Compare modes on matched questions, report coverage, and retain missing-score reasons. |
| RAPTOR has high relevancy but much lower context recall; R2 abstains on an answerable question. | Topical answers and relevant-sounding responses do not establish that the needed evidence was retrieved. | Label answer-bearing passages and measure retrieval recall at a fixed passage budget before generation. |
| KG returns no context on 76.50% of questions. | Retrieval availability is a major observed bottleneck; a better generator alone cannot supply missing evidence. | Audit empty KG queries and compare KG with a fallback to book retrieval. |
| G1 and G2 fail despite supporting evidence in the first passage. | Retrieval improvements alone will not resolve relation and entity confusion. | Evaluate generation with manually verified contexts and check each answer claim against its entity and relation. |
| R2 scores perfectly on two metrics; G2 receives zero context recall despite visible support. | Automated scores need manual inspection and an explicit measure of answer correctness or task completion. | Review disagreements with the evaluator and record correctness, abstention, and source support separately. |

For the next evaluation, annotate **retrieval sufficiency** first, then assess **generation correctness and source support** conditional on that evidence. Keep appropriate abstentions separate from unsupported assertions, and permit a mixed label when both stages fail. This would distinguish whether a change improves evidence selection or the use of already sufficient evidence.

## Scope and reproducibility

All four saved QA files contain 400 aligned questions and references, and the per-question CSVs align with those logs. The principal results above reproduce the committed summaries; context-return counts and the shared-valid faithfulness comparison were recomputed directly from the saved files.

The QA artifacts store only `question`, `answer`, `contexts`, and `ground_truth`. They do not record the selected route for Routed queries, document identifiers, the exact prompt, model revisions, effective run settings, or per-sample execution errors. The CSVs also omit evaluator configuration and missing-score reasons. Current code defaults therefore do not establish how the historical runs were configured, and these records cannot isolate a router-specific cause or a particular retrieval component's contribution.

The report does not claim statistical significance, full error-category frequencies, or an overall accuracy score. Future runs should save this provenance and a manually reviewed correctness set alongside the Ragas metrics.
