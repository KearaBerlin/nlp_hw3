from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch
# import math

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.generation_config.pad_token_id = model.generation_config.eos_token_id

prompt = "Roses are red, violets are"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

def generate_sentence(description, do_sample=False, num_beams=1, top_k=50, top_p=1):
    output = model.generate(input_ids, do_sample=do_sample, 
                                      num_beams=num_beams, top_k=top_k,
                                      top_p=top_p, max_length=30,
                                      pad_token_id=tokenizer.eos_token_id,
                                      output_scores=True,
                                      return_dict_in_generate=True)
    tokens = output.sequences[0]
    # tokens = output[0]
    sentence = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    print(f"{description}: {''.join(sentence)}")

    # we tried this method of getting output logits instead of using output_scores=True in model.generate,
    # and it gave non-zero probabilities for the sampling decoding methods, but the perplexities ended
    # up being several orders of magnitude too large still (~1e34).
    # with torch.no_grad():
    #     outputs = model(tokens)

    # we also tried using compute_transition_scores instead of getting all logits and choosing the correct
    # one manually, but we ran into the same issues.

    # possibly the issue is with how we are passing parameters to model.generate() in the first place.

    softmax = torch.nn.Softmax(dim=-1)
    token_probs = []
    for (token_id, logits) in zip(tokens, output.scores):
    # for (token_id, logits) in zip(tokens, outputs.logits):
        probs = softmax(logits)
        token_probs.append(probs[0,token_id].item())
        # token_probs.append(probs[token_id].item())
    token_probs = torch.tensor(token_probs)

    likelihood = sum(torch.log(token_probs))

    N = len(tokens)
    perplexity = torch.exp(-1/N * likelihood)

    # double checking the calculation is the same with a different base
    # likelihood_2 = 0
    # for p in token_probs.numpy():
    #     likelihood_2 += math.log(p,2)
    # perplexity_2 = math.pow(2, -1/N * likelihood_2)

    print(f"Likelihood: {likelihood} Perplexity: {perplexity}")

# greedy search (num_beams=1 (default) and do_sample=False)
# generate up to 30 tokens
generate_sentence("greedy search", do_sample=False)

# beam search (num_beams > 1, do_sample = False)
# https://huggingface.co/blog/how-to-generate suggests 5 beams
generate_sentence("beam search 5 beams", do_sample=False, num_beams=5)

# top-k sampling (do_sample = True, top_p=1)
# default k is 50
generate_sentence("top-k sampling, k=50", do_sample=True, top_k=50)
generate_sentence("top-k sampling, k=5", do_sample=True, top_k=5)

# top-p sampling (do_sample = True, top_p < 1)
# p = 0.75 from class slides
generate_sentence("top-p sampling, p=0.75", do_sample=True, top_p=0.75, top_k=0)
