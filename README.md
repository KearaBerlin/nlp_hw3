# nlp_hw3

## Install required packages:

pip install torch

pip install transformers

pip install evaluate

pip install bert_score

pip install rouge_score

## To run sentence generation with GPT-2:

python main.py

## To run document summarization from Gigaword dataset:

python summarization.py

## Description

This repository contains code for Homework 3 from Natural Language Processing at the University of Minnesota
in Spring 2023.
Downloads GPT-2 pretrained language model and uses 4 decoding methods to 
generate a sentence from a prompt; uses bart-large-gigaword model to perform document summarization on the Gigaword dataset.

https://huggingface.co/a1noack/bart-large-gigaword

https://huggingface.co/datasets/gigaword

We tried to ask ChatGPT how to calculate perplexity of a language model, and it gave us the code in chatgpt.py, but we did not end up incorporating this code into the rest of the project.
