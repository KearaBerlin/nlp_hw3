from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("a1noack/bart-large-gigaword")
model = AutoModel.from_pretrained("a1noack/bart-large-gigaword")

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

for (document, summary) in zip(documents, summaries):
    print(f"\n----------------\n{document}")
    input_ids = tokenizer(document, return_tensors="pt").input_ids

    # greedy search (num_beams=1 (default) and do_sample=False)
    # generate up to 30 tokens
    generate_sentence("greedy search", input_ids, do_sample=False)

    # beam search (num_beams > 1, do_sample = False)
    # https://huggingface.co/blog/how-to-generate suggests 5 beams
    generate_sentence("beam search 5 beams", input_ids, do_sample=False, num_beams=5)

    # top-k sampling (do_sample = True, top_p=1)
    # default k is 50
    generate_sentence("top-k sampling, k=5", input_ids, do_sample=True, top_k=5)

    # top-p sampling (do_sample = True, top_p < 1)
    # p = 0.75 from class slides
    generate_sentence("top-p sampling, p=0.75", input_ids, do_sample=True, top_p=0.75, top_k=0)