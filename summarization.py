from datasets import load_dataset
from evaluate import load
from transformers import BartForCausalLM, AutoTokenizer
import csv

tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
model = BartForCausalLM.from_pretrained("a1noack/bart-large-gigaword")

bertscore = load("bertscore")
rouge = load('rouge')

dataset = load_dataset("gigaword", split="test")
documents = dataset['document'][:50]
summaries = dataset['summary'][:50]

def generate_sentence(description, input_ids, do_sample=False, num_beams=1, top_k=50, top_p=1):
    output = model.generate(input_ids, do_sample=do_sample, 
                                      num_beams=num_beams, top_k=top_k,
                                      top_p=top_p, max_length=30,
                                      pad_token_id=tokenizer.eos_token_id)
                                    #   output_scores=True,
                                    #   return_dict_in_generate=True)
    # tokens = output.sequences[0]
    tokens = output[0]
    sentence = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    s = "".join(sentence)
    print(f"{description}: {s}")
    return s

def calculate_metrics(sentence, reference):
    bert_results = bertscore.compute(predictions=[sentence], references=[reference], lang="en")
    rouge_results = rouge.compute(predictions=[sentence], references=[reference])
    return bert_results, rouge_results

for (document, summary) in zip(documents, summaries):
    print(f"\n----------------\n{document}")
    print(f"Summary: {summary}\n")

    input_ids = tokenizer(document, return_tensors="pt").input_ids
    summary_tokens = tokenizer(summary, return_tensors="pt").input_ids

    # greedy search (num_beams=1 (default) and do_sample=False)
    # generate up to 30 tokens
    s1 = generate_sentence("greedy search", input_ids, do_sample=False)
    results = calculate_metrics(s1, summary)

    # beam search (num_beams > 1, do_sample = False)
    # https://huggingface.co/blog/how-to-generate suggests 5 beams
    s2 = generate_sentence("beam search 5 beams", input_ids, do_sample=False, num_beams=5)
    results = calculate_metrics(s2, summary)

    # top-k sampling (do_sample = True, top_p=1)
    # default k is 50
    s3 = generate_sentence("top-k sampling, k=5", input_ids, do_sample=True, top_k=5)
    results = calculate_metrics(s3, summary)

    # top-p sampling (do_sample = True, top_p < 1)
    # p = 0.75 from class slides
    s4 = generate_sentence("top-p sampling, p=0.75", input_ids, do_sample=True, top_p=0.75, top_k=0)
    results = calculate_metrics(s4, summary)

    with open('output.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([document, summary, s1, s2, s3, s4])