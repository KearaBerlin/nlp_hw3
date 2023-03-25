from datasets import load_dataset
from evaluate import load
from transformers import BartForConditionalGeneration, AutoTokenizer
import csv

tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
model = BartForConditionalGeneration.from_pretrained("a1noack/bart-large-gigaword")

bertscore = load("bertscore")
rouge = load('rouge')

dataset = load_dataset("gigaword", split="test")
documents = dataset['document'][:50]
summaries = dataset['summary'][:50]

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Sentence', 'Summary', 'Greedy Search', 'Greedy Search Bert Score', 'Greedy Search Rouge Score', 'Beam Search', 'Beam Search Bert Score', 'Beam Search Rouge Score', 'Top-K', 'Top-K Bert Score', 'Top-K Rouge Score', 'Top-P', 'Top-P Bert Score', 'Top-P Rouge Score'])

def generate_sentence(description, input_ids, do_sample=False, num_beams=1, top_k=50, top_p=1):
    output = model.generate(input_ids, do_sample=do_sample,
                                      num_beams=num_beams, top_k=top_k,
                                      top_p=top_p, min_length=0, max_new_tokens=12,
                                      pad_token_id=tokenizer.eos_token_id)
    tokens = output[0]

    sentence = tokenizer.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    s = "".join(sentence)
    print(f"{description}: {s}")
    return s

def calculate_metrics(sentence, reference):
    bert_results = bertscore.compute(predictions=[sentence], references=[reference], lang="en")
    rouge_results = rouge.compute(predictions=[sentence], references=[reference])
    return bert_results['f1'][0], rouge_results['rouge1']

for (document, summary) in zip(documents, summaries):
    print(f"\n----------------\n{document}")
    print(f"Summary: {summary}\n")

    input_ids = tokenizer(document, return_tensors="pt", truncation=True, max_length=128,padding=True)['input_ids']
    summary_tokens = tokenizer(summary, return_tensors="pt").input_ids

    # greedy search (num_beams=1 (default) and do_sample=False)
    # generate up to 30 tokens
    s1 = generate_sentence("greedy search", input_ids, do_sample=False)
    bert_results1, rouge_result1 = calculate_metrics(s1, summary)

    # beam search (num_beams > 1, do_sample = False)
    # https://huggingface.co/blog/how-to-generate suggests 5 beams
    s2 = generate_sentence("beam search 5 beams", input_ids, do_sample=False, num_beams=5)
    bert_results2, rouge_result2 = calculate_metrics(s2, summary)

    # top-k sampling (do_sample = True, top_p=1)
    # default k is 50
    s3 = generate_sentence("top-k sampling, k=5", input_ids, do_sample=True, top_k=5)
    bert_results3, rouge_result3 = calculate_metrics(s3, summary)

    # top-p sampling (do_sample = True, top_p < 1)
    # p = 0.75 from class slides
    s4 = generate_sentence("top-p sampling, p=0.75", input_ids, do_sample=True, top_p=0.75, top_k=0)
    bert_results4, rouge_result4 = calculate_metrics(s4, summary)

    with open('output.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([document, summary, s1, bert_results1, rouge_result1, s2, bert_results2, rouge_result2, s3, bert_results3, rouge_result3, s4, bert_results4, rouge_result4])
