# Conversion

The following example demonstrates how to convert a GPTQ model from the HF hub to Marlin format.

### Install

In addition to Marlin and PyTorch, install the following:

```bash
pip install -U transformers accelerate auto-gptq optimum
```

### Convert GTPQ Model to Marlin Format

The following converts the model from GPTQ to Marlin format. Note that this requires:
- `sym=true`
- `group_size=128`
-  `desc_activations=false`

```bash
python3 convert.py --model-id "TheBloke/Llama-2-7B-Chat-GPTQ" --save-path "./marlin-model" --do-generation
```

### Load Marlin Model

The following loads the Marlin model from disk.

```python
from load import load_model
from transformers import AutoTokenizer

# Load model from disk.
model_path = "./marlin-model"
model = load_model(model_path).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(model_path)


# Run inference to confirm it is working.
inputs = tokenizer("My favorite song is", return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.batch_decode(outputs)[0])
```

Output:
```bash
<s> My favorite song is "Bohemian Rhapsody" by Queen. I love the operatic vocals, the guitar solo, and the way the song builds from a slow ballad to a full-on rock anthem. I've been listening to it
```