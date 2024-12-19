import json
import transformers
import torch
import os
import sys
from jarvis.io.vasp.inputs import Poscar
from pydantic_settings import BaseSettings
from typing import Optional

class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""
    id_prop_path: Optional[str] = "robo_desc.json.zip"
    prefix: str = "atomgpt_run"
    model_name: str = "gpt2"
    batch_size: int = 16
    max_length: int = 512
    num_epochs: int = 500
    latent_dim: int = 1024
    learning_rate: float = 1e-3
    test_each_run: bool = True
    include_struct: bool = False
    pretrained_path: str = ""
    seed_val: int = 42
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    output_dir: str = "out_temp"
    train_ratio: Optional[float] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    keep_data_order: bool = True
    desc_type: str = "desc_2"
    convert: bool = False

def main():
    # Parse command-line arguments
    if len(sys.argv) != 3:
        print("Usage: python forward_prediction_script.py <config.json> <POSCAR>")
        sys.exit(1)
    
    config_file_path = sys.argv[1]
    poscar_file_path = sys.argv[2]

    # Load the configuration file
    with open(config_file_path, "r") as f:
        config_data = json.load(f)

    config = TrainingPropConfig(**config_data)
    print("Loaded configuration:")
    print(config)

    # Prepare device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model_name = config.model_name
    output_dir = config.output_dir

    if "t5" in model_name:
        model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
        )

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    model.lm_head = torch.nn.Sequential(
        torch.nn.Linear(model.config.hidden_size, config.latent_dim),
        torch.nn.Linear(config.latent_dim, 1),
    )

    # Load model weights
    state_dict = torch.load(os.path.join(output_dir, "best_model.pt"), map_location=device)
    # Remove 'module.' prefix if present
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    # Load input POSCAR
    atoms = Poscar.from_file(poscar_file_path).atoms
    print("Inputted Atoms:")
    print(atoms)

    # Generate descriptor
    desc = atoms.describe()[config.desc_type]

    # Encode input POSCAR string
    with open(poscar_file_path, "r") as f:
        pos_str = f.read()

    max_length = config.max_length
    input_ids = tokenizer(
        [pos_str],
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )["input_ids"]

    input_ids = input_ids.to(device)
    model = model.to(device)

    # Make prediction
    if "t5" in model_name:
        predictions = (
            model(
                input_ids,
                decoder_input_ids=input_ids,
            )
            .logits.squeeze()
            .mean(dim=-1)
        )
    else:
        predictions = (
            model(
                input_ids,
            )
            .logits.squeeze()
            .mean(dim=-1)
        )

    predictions = predictions.cpu().detach().numpy().tolist()
    print("Predicted bandgap:")
    print(predictions)

if __name__ == "__main__":
    main()
