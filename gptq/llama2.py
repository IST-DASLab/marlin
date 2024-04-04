import time
from argparse import Namespace
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from gptq import *
from quant import *
import marlin


DEV = torch.device("cuda:0")


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=""):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


def maybe_0th_element(x):
    if isinstance(x, Sequence):
        return x[0]
    return x


def get_llama(name):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM

    model = LlamaForCausalLM.from_pretrained(name, torch_dtype="auto")
    model.config.pretraining_tp = 1
    model.seqlen = 4096
    return model


def make_batch_iterator(*tensor_lists, batch_size: int):
    all_batch_indices = []
    dataset_size = len(tensor_lists[0])
    while True:
        if len(all_batch_indices) == 0:
            all_batch_indices = list(torch.randperm(dataset_size).chunk(dataset_size // batch_size))
        batch_indices = all_batch_indices.pop(0)
        yield [torch.cat([tensor_list[i] for i in batch_indices], dim=0) for tensor_list in tensor_lists]


@torch.enable_grad()
def finetune_block(
    layer: nn.Module,
    inps: List[torch.Tensor],
    attention_masks: List[torch.Tensor],
    position_ids: List[torch.Tensor],
    targets: List[torch.Tensor],
    args: Namespace,
):
    print("Finetuning ...")
    dtype = next(layer.parameters()).dtype
    layer.train()
    # cast to float32
    layer.float()

    steps_per_epoch = len(inps) // args.batch_size

    batch_iterator = make_batch_iterator(inps, attention_masks, position_ids, targets, batch_size=args.batch_size)
    # init optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=args.lr)
    # init scaler
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    for epoch in range(args.finetune_epochs):
        epoch_loss = 0
        for step in range(steps_per_epoch):
            inps_, attention_mask_, position_ids_, targets_ = next(batch_iterator)
            inps_, targets_, attention_mask_ = inps_.float(), targets_.float(), attention_mask_.float()
            with torch.autocast(device_type="cuda", enabled=args.amp):
                out = maybe_0th_element(layer(inps_, attention_mask=attention_mask_, position_ids=position_ids_))
            loss = F.mse_loss(out, targets_)
            # scaler and optimizer step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            epoch_loss += loss.item() / steps_per_epoch
        print(f"Epoch {epoch}. Train loss: {epoch_loss:.2e}")

    layer = layer.to(dtype)
    layer.eval()


@torch.no_grad()
def llama_sequential(model, dataloader, dev, args):
    print("Starting ...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    print("Ready.")
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        # prepare outs if finetuning
        if args.finetune_epochs:
            outs = []

        for names in sequential:
            if model.config.num_attention_heads != model.config.num_key_value_heads and args.skip_gq:
                names.remove("self_attn.k_proj")
                names.remove("self_attn.v_proj")

            subset = {n: full[n] for n in names}

            gptq = {}
            for name in subset:
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = Quantizer()
                gptq[name].quantizer.configure(args.wbits)

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                out = maybe_0th_element(layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j]))
                if args.finetune_epochs:
                    outs.append(out)
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Quantizing ...")
                qweight, scale, zero = gptq[name].fasterquant(
                    percdamp=args.percdamp, groupsize=args.groupsize, clip=not args.no_clip, baseline=args.nearest
                )
                qlayer = QLinear(
                    qweight,
                    scale,
                    zero,
                    bias=gptq[name].layer.bias,
                )
                parent_name, child_name = name.rsplit(".", 1)
                parent_module = layer.get_submodule(parent_name)
                setattr(parent_module, child_name, qlayer)
            # tune block
            if args.finetune_epochs:
                finetune_block(layer, inps, attention_masks, position_ids, outs, args)
                torch.cuda.empty_cache()

        for j in range(args.nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]

        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()

    model.config.use_cache = use_cache


@torch.no_grad()
def llama_eval(model, dataloader, dev):
    print("Evaluating ...")

    nsamples = len(dataloader)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = []
    attention_masks = []
    position_ids = []

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            attention_masks.append(kwargs["attention_mask"])
            position_ids.append(kwargs["position_ids"])
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch.to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    for i in range(len(layers)):
        print(f"Layer {i}")
        layer = layers[i].to(dev)
        for j in range(nsamples):
            inps[j] = layer(inps[j], attention_mask=attention_masks[j], position_ids=position_ids[j])[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i]
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = (dataloader[i].to(dev))[:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item()}")

    model.config.use_cache = use_cache


def llama_pack(model):
    layers = find_layers(model, [QLinear])
    marlin.replace_linear(model, lambda n: n in layers, groupsize=args.groupsize, layers=(nn.Linear, QLinear))
    qlayers = find_layers(model, [marlin.Layer])
    print("Packing ...")
    for name in qlayers:
        print(name)
        qlayers[name].pack(layers[name].to(DEV))
        qlayers[name].cpu()
        layers[name].cpu()
    print("Done.")
    return model


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="LlaMa model to load; pass location of hugginface converted checkpoint."
    )
    parser.add_argument("--dataset", type=str, default="red", help="Where to extract calibration data from.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for sampling the calibration data.")
    parser.add_argument("--nsamples", type=int, default=256, help="Number of calibration data samples.")
    parser.add_argument(
        "--percdamp", type=float, default=0.1, help="Percent of the average Hessian diagonal to use for dampening."
    )
    parser.add_argument("--nearest", action="store_true", help="Whether to run the RTN baseline.")
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        choices=[-1, 128],
        help="Groupsize to use for quantization; default is 128.",
    )
    parser.add_argument("--true-sequential", action="store_true", help="Whether to run in true sequential model.")
    parser.add_argument(
        "--no_clip", action="store_true", help="Whether to skip hessian based grid clipping when using groups."
    )
    parser.add_argument(
        "--skip_gq",
        action="store_true",
        help="Whether to skip quantizing group keys and values for the 70B model with group-query attention.",
    )
    parser.add_argument("--save", type=str, default="", help="Whether and where to save the quantized model.")
    # evaluation params
    parser.add_argument(
        "--eval_datasets", nargs="+", type=str, default=["wikitext2", "red"], help="Datasets for perplexity evaluation."
    )
    # finetuning params
    parser.add_argument("--lr", type=float, default=1e-5, help="Finetuning learning rate.")
    parser.add_argument("--batch_size", type=int, default=1, help="Finetuning batch size.")
    parser.add_argument(
        "--finetune_epochs", type=int, default=0, help="How many epochs to finetune the block after quantization."
    )
    parser.add_argument("--amp", action="store_true", help="Whether to use amp on block finetuning.")

    args = parser.parse_args()

    if args.nearest:
        args.nsamples = 0

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16:
        tick = time.perf_counter()
        llama_sequential(model, dataloader, DEV, args)
        print(f"Quantization took: {(time.perf_counter() - tick):.2f} s.")

    for dataset in args.eval_datasets:
        dataloader, testloader = get_loaders(dataset, seed=args.seed, model=args.model, seqlen=model.seqlen)
        print(f"Dataset: {dataset}")
        llama_eval(model, testloader, DEV)

    if args.save:
        args.save += ".marlin"
        if args.groupsize != -1:
            args.save += ".g%d" % args.groupsize
        llama_pack(model)
        torch.save(model.state_dict(), args.save)
