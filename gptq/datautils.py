import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    import random
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
        testloader.append(testenc.input_ids[:, i:(i + seqlen)])

    return trainloader, testloader 

def get_red(nsamples, seed, seqlen, model):
    VALSAMPLES = 1024

    from datasets import load_dataset
    from transformers import AutoTokenizer 
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    traindata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')

    np.random.seed(0)
    perm = np.random.permutation(len(traindata))

    dataloader = []
    for i in perm:
        tokens = tokenizer(traindata[int(i)]['text'], return_tensors='pt').input_ids
        if not (1 < tokens.shape[1] <= seqlen):
            continue
        dataloader.append(tokens)
        if len(dataloader) == nsamples + VALSAMPLES:
            break
    trainloader = dataloader[VALSAMPLES:]
    testloader = dataloader[:VALSAMPLES]
    return trainloader, testloader


def get_loaders(
    name, nsamples=256, seed=0, seqlen=2048, model=''
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model)
        return data, None
    if 'red' in name:
        return get_red(nsamples, seed, seqlen, model)

