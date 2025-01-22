from jarvis.db.jsonutils import loadjson
from typing import Optional
from atomgpt.inverse_models.loader import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from jarvis.core.atoms import Atoms
from jarvis.db.jsonutils import loadjson, dumpjson
import numpy as np
from jarvis.core.lattice import Lattice
from tqdm import tqdm
import pprint
from jarvis.io.vasp.inputs import Poscar
import csv
import os
from pydantic_settings import BaseSettings
import sys
import argparse


parser = argparse.ArgumentParser(
    description="Atomistic Generative Pre-trained Transformer."
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)


class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    id_prop_path: Optional[str] = "id_prop.csv"
    prefix: str = "atomgpt_run"
    model_name: str = "unsloth/mistral-7b-bnb-4bit"
    batch_size: int = 2
    num_epochs: int = 2
    seed_val: int = 42
    num_train: Optional[int] = 2
    num_val: Optional[int] = 2
    num_test: Optional[int] = 2
    model_save_path: str = "lora_model_m"


# The list of models you might try. We'll pick the last one for local usage.
fourbit_models = [
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.2-bnb-4bit",
    "unsloth/llama-2-7b-bnb-4bit",
    "unsloth/llama-2-13b-bnb-4bit",
    "unsloth/codellama-34b-bnb-4bit",
    "unsloth/tinyllama-bnb-4bit",
    "/users/crc00042/mistral-7b-bnb-4bit",
]
nm = fourbit_models[-1]  # points to your local model folder

instruction = "Below is a description of a superconductor material."
alpaca_prompt = """Below is a description of a superconductor material..

### Instruction:
{}

### Input:
{}

### Output:
{}"""


def get_crystal_string_t(atoms):
    lengths = atoms.lattice.abc
    angles = atoms.lattice.angles
    atom_ids = atoms.elements
    frac_coords = atoms.frac_coords

    crystal_str = (
        " ".join(["{0:.2f}".format(x) for x in lengths])
        + "\n"
        + " ".join([str(int(x)) for x in angles])
        + "\n"
        + "\n".join(
            [
                str(t) + " " + " ".join(["{0:.3f}".format(xx) for xx in c])
                for t, c in zip(atom_ids, frac_coords)
            ]
        )
    )
    return crystal_str


def make_alpaca_json(dataset=[], jids=[], prop="Tc_supercon", include_jid=False):
    mem = []
    for i in dataset:
        if i[prop] != "na" and i["id"] in jids:
            atoms = Atoms.from_dict(i["atoms"])
            info = {}
            if include_jid:
                info["id"] = i["id"]
            info["instruction"] = instruction
            info["input"] = (
                "The chemical formula is "
                + atoms.composition.reduced_formula
                + ". The  "
                + prop
                + " is "
                + str(round(i[prop], 3))
                + "."
                + " Generate atomic structure description with lattice lengths, angles, coordinates and atom types."
            )
            info["output"] = get_crystal_string_t(atoms)
            mem.append(info)
    return mem


def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    EOS_TOKEN = "</s>"
    for instruction, inp, out in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise generation can go on forever!
        text = alpaca_prompt.format(instruction, inp, out) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}


def text2atoms(response):
    """
    Attempt to parse the lines in `response`:
    line 1: (optional blank)
    line 2: lattice lengths
    line 3: lattice angles
    subsequent lines: atom symbol + x y z

    Return an Atoms object or None if parsing fails.
    """
    lines = response.strip().split("\n")
    print("[DEBUG] text2atoms got these lines:")
    for idx, line in enumerate(lines):
        print(f"  line {idx}: {line}")

    if len(lines) < 3:
        print("[DEBUG] Not enough lines for lat lengths/angles. Returning None.")
        return None

    try:
        # line 1 might be blank or something else. We will assume line 1 has lengths, line 2 angles, etc.
        lat_lengths = np.array(lines[0].split(), dtype="float")
        lat_angles = np.array(lines[1].split(), dtype="float")
        if lat_lengths.size != 3 or lat_angles.size != 3:
            print(f"[DEBUG] lat_lengths or lat_angles not size 3: {lat_lengths}, {lat_angles}")
            return None
        # The rest are elements + coords
        elements = []
        coords = []
        for ii in range(2, len(lines)):
            split_line = lines[ii].split()
            if len(split_line) != 4:
                print("[DEBUG] Skipping line because it doesn't have 4 tokens:", lines[ii])
                continue
            elements.append(split_line[0])
            try:
                c = [float(x) for x in split_line[1:4]]
            except Exception as e:
                print("[DEBUG] Could not convert coords to float:", e)
                continue
            coords.append(c)

        if len(elements) == 0:
            print("[DEBUG] No valid atom lines found. Returning None.")
            return None

        lat = Lattice.from_parameters(
            lat_lengths[0], lat_lengths[1], lat_lengths[2],
            lat_angles[0], lat_angles[1], lat_angles[2]
        )

        atoms = Atoms(coords=coords, elements=elements, lattice_mat=lat.lattice(), cartesian=False)
        return atoms

    except Exception as exp:
        print("[DEBUG] Exception while parsing text2atoms:", exp)
        return None


def gen_atoms(prompt="", max_new_tokens=512, model=None, tokenizer=None):
    """
    Generate text for the given prompt, then parse it into an Atoms object.
    Return the Atoms object or None if parsing fails.
    """
    if model is None or tokenizer is None:
        print("[DEBUG] No model/tokenizer provided to gen_atoms. Returning None.")
        return None

    # Build the full prompt
    final_prompt = alpaca_prompt.format(instruction, prompt, "")
    print(f"[DEBUG] gen_atoms final_prompt:\n{final_prompt}")

    inputs = tokenizer([final_prompt], return_tensors="pt").to("cuda")

    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True
    )

    # Print the raw generation for debugging
    raw_text = tokenizer.batch_decode(outputs)[0]
    print("[DEBUG] Full model output:\n", raw_text)

    # Attempt to find '# Output:' tag
    if "# Output:" in raw_text:
        print("[DEBUG] Found '# Output:' in model output.")
        split_by_output = raw_text.split("# Output:")
        if len(split_by_output) < 2:
            print("[DEBUG] '# Output:' found but split has < 2 elements:", split_by_output)
            # Fall back to entire text
            response = split_by_output[-1]
        else:
            response = split_by_output[1]
    else:
        print("[DEBUG] '# Output:' not found. Using entire generation as response.")
        response = raw_text

    # Now parse the result into Atoms
    atoms = text2atoms(response)
    return atoms


def run_atomgpt_inverse(config_file="config.json"):
    run_path = os.path.abspath(config_file).split("config.json")[0]
    config = loadjson(config_file)
    config = TrainingPropConfig(**config)
    pprint.pprint(config)

    id_prop_path = config.id_prop_path
    num_train = config.num_train
    num_test = config.num_test

    # Construct the path to the CSV
    id_prop_csv = os.path.join(run_path, id_prop_path)
    print(f"[DEBUG] Loading CSV from: {id_prop_csv}")
    with open(id_prop_csv, "r") as f:
        reader = csv.reader(f)
        dt = [row for row in reader]

    # Build the dataset
    dat = []
    ids = []
    for row in tqdm(dt, desc="Reading id_prop.csv"):
        info = {}
        info["id"] = row[0]
        ids.append(row[0])
        info["prop"] = float(row[1])
        # Construct path to POSCAR
        poscar_path = os.path.join(run_path, info["id"])
        print(f"[DEBUG] Loading POSCAR from: {poscar_path}")
        atoms = Atoms.from_poscar(poscar_path)
        info["atoms"] = atoms.to_dict()
        dat.append(info)

    # Split train/test
    train_ids = ids[0:num_train]
    test_ids = ids[num_train:]

    # Create Alpaca JSON
    m_train = make_alpaca_json(dataset=dat, jids=train_ids, prop="prop")
    dumpjson(data=m_train, filename="alpaca_prop_train.json")

    m_test = make_alpaca_json(
        dataset=dat, jids=test_ids, prop="prop", include_jid=True
    )
    dumpjson(data=m_test, filename="alpaca_prop_test.json")

    # Model settings
    max_seq_length = 2048
    dtype = None
    load_in_4bit = True

    # Load the base model
    print(f"[DEBUG] from_pretrained with local model name: {nm}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=nm,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    print("[DEBUG] Model loaded successfully.")

    # Convert to a LoRA trainable model
    print("[DEBUG] Converting base model to LoRA model.")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    print("[DEBUG] LoRA model ready.")

    # Prepare training data
    dataset = load_dataset("json", data_files="alpaca_prop_train.json", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True, num_proc=1)

    # Setup trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            overwrite_output_dir=True,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
            num_train_epochs=config.num_epochs,  # or 5 as in your example
            report_to="none",
        ),
    )

    print("[DEBUG] Starting training.")
    trainer_stats = trainer.train()
    print("[DEBUG] Training complete:", trainer_stats)

    print(f"[DEBUG] Saving PEFT model to {config.model_save_path}")
    model.save_pretrained(config.model_save_path)

    # Reload the fine-tuned model for inference
    print(f"[DEBUG] Reloading the model from {config.model_save_path} for testing.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model_save_path,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    print("[DEBUG] Model is ready for inference.")

    # Evaluate on test set
    out_file = "AI-AtomGen-prop-dft_3d-test-rmse.csv"
    print(f"[DEBUG] Writing test results to {out_file}")
    f = open(out_file, "w")
    f.write("id,target,prediction\n")

    for i in tqdm(m_test, desc="Evaluating on test set"):
        prompt = i["input"]
        print(f"[DEBUG] Test prompt for ID={i['id']}: {prompt}")

        gen_mat = gen_atoms(prompt=prompt, tokenizer=tokenizer, model=model)
        if gen_mat is None:
            print("[DEBUG] gen_atoms returned None. Skipping writing output for this entry.")
            f.write(f"{i['id']},{i['output']},None\n")
            continue

        target_mat = text2atoms("\n" + i["output"])
        print("[DEBUG] target_mat:", target_mat)
        print("[DEBUG] genmat:", gen_mat)

        # If you really want to write them to CSV as POSCAR:
        # But check if target_mat or gen_mat is None
        if target_mat is not None and gen_mat is not None:
            line = (
                i["id"]
                + ","
                + Poscar(target_mat).to_string().replace("\n", "\\n")
                + ","
                + Poscar(gen_mat).to_string().replace("\n", "\\n")
                + "\n"
            )
            f.write(line)
        else:
            f.write(f"{i['id']},None,None\n")

        print()  # blank line

    f.close()
    print("[DEBUG] Finished evaluation. CSV saved.")


def main():
    args = parser.parse_args(sys.argv[1:])
    run_atomgpt_inverse(config_file=args.config_name)


if __name__ == "__main__":
    main()
