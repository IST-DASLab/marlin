import os
import random

import numpy as np
import torch

from datasets import load_dataset
from transformers import AutoTokenizer


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    testloader = []
    for i in range(0, testenc.input_ids.shape[1] - seqlen, seqlen):
        testloader.append(testenc.input_ids[:, i : (i + seqlen)])

    return trainloader, testloader


def get_red(nsamples, seed, seqlen, model):
    VALSAMPLES = 1024

    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train")

    np.random.seed(seed)
    perm = np.random.permutation(len(traindata))

    dataloader = []
    for i in perm:
        tokens = tokenizer(traindata[int(i)]["text"], return_tensors="pt").input_ids
        if not (1 < tokens.shape[1] <= seqlen):
            continue
        dataloader.append(tokens)
        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    return trainloader, testloader


def get_dolly_hhrlhf(nsamples, seed, seqlen, model):
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset("mosaicml/dolly_hhrlhf", split="train")
    dataiter = iter(traindata)

    random.seed(seed)

    trainloader = []
    for _ in range(nsamples):
        cur_seqlen = 0
        trainencs = []
        while True:
            sample = next(dataiter)
            raw_seq = sample["prompt"] + sample["response"] + tokenizer.eos_token
            trainencs.append(tokenizer(raw_seq, return_tensors="pt").input_ids)
            cur_seqlen += trainencs[-1].numel()
            if cur_seqlen > seqlen:
                break
        trainenc = torch.cat(trainencs, dim=-1)
        i = random.randint(0, trainenc.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc[:, i:j]
        assert inp.shape[1] == seqlen
        trainloader.append(inp)
    return trainloader, None


def get_loaders(name, nsamples=256, seed=0, seqlen=2048, model: str = ""):
    if os.path.isfile(name):
        samples = torch.load(name)[:nsamples]
        assert isinstance(samples, list)
        assert isinstance(samples[0], torch.Tensor)
        return [sample[..., :seqlen] for sample in samples], None
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
    if "red" in name:
        return get_red(nsamples, seed, seqlen, model)
    if "dolly_hhrlhf" in name:
        return get_dolly_hhrlhf(nsamples, seed, seqlen, model)
    else:
        raise ValueError("Unknown dataset")
