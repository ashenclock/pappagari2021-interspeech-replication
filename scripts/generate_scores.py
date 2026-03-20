"""Generate OOF and test scores for all models listed in config.yaml."""

import argparse
import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.data import get_data_splits, get_dataloaders
from src.models import build_model
from src.utils import set_seed, clear_memory, load_labels_df


def _recursive_update(base_dict, update_dict):
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
            base_dict[key] = _recursive_update(base_dict[key], value)
        else:
            base_dict[key] = value
    return base_dict


def _get_scores(model, loader, device, task):
    model.eval()
    all_scores, all_ids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inferencing", leave=False):
            ids = batch.pop("id")
            all_ids.extend(ids)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            scores = torch.softmax(outputs, dim=1)[:, 1] if task == 'classification' else outputs
            all_scores.extend(scores.cpu().numpy())
    return all_ids, all_scores


def process_single_model(model_info: dict, base_config: Config):
    output_dir = Path(model_info['output_dir'])
    print(f"\n{'='*20} Processing: {output_dir.name} {'='*20}")

    if not output_dir.exists():
        print(f"WARNING: '{output_dir}' does not exist. Skipping.")
        return

    model_config_dict = copy.deepcopy(base_config.to_dict())
    model_config_dict = _recursive_update(model_config_dict, model_info)

    if model_info.get('modality') == 'text':
        potential_transcript_folder = model_info['output_dir'].split('_')[-1]
        model_config_dict['transcription_model_for_training'] = potential_transcript_folder

    model_config = Config(model_config_dict)
    device = torch.device(model_config.device)
    clear_memory()

    # OOF predictions
    print("\n[1/2] Generating OOF predictions...")
    oof_predictions = []
    for fold, train_df, val_df, _ in get_data_splits(model_config):
        model_path = output_dir / f"fold_{fold}" / "best_model.pt"
        if not model_path.exists():
            continue
        model = build_model(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, val_loader = get_dataloaders(model_config, train_df, val_df)
        val_ids, val_scores = _get_scores(model, val_loader, device, model_config.task)
        oof_predictions.append(pd.DataFrame({'ID': val_ids, 'score': val_scores}))

    if oof_predictions:
        full_oof_df = pd.concat(oof_predictions).set_index('ID')
        train_labels_df = load_labels_df(model_config.data.train_labels).set_index('ID')
        final_oof_df = full_oof_df.join(train_labels_df)
        final_oof_df.to_csv(output_dir / "oof_predictions.csv")
        print("  -> OOF predictions saved.")

    # Test predictions
    print("\n[2/2] Generating test predictions...")
    all_test_scores = []
    _, _, _, test_df = next(get_data_splits(model_config))
    _, _, test_loader = get_dataloaders(model_config, test_df, test_df, test_df=test_df)
    test_ids = []

    for fold in range(model_config.k_folds):
        model_path = output_dir / f"fold_{fold}" / "best_model.pt"
        if not model_path.exists():
            continue
        model = build_model(model_config).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        ids, scores = _get_scores(model, test_loader, device, model_config.task)
        if not test_ids:
            test_ids = ids
        all_test_scores.append(scores)

    if all_test_scores:
        mean_scores = np.mean(all_test_scores, axis=0)
        pd.DataFrame({'ID': test_ids, 'score': mean_scores}).to_csv(output_dir / "test_predictions.csv", index=False)
        print("  -> Test predictions saved.")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for all models in config.yaml.")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    main_config = load_config(args.config)
    set_seed(main_config.seed)

    if not hasattr(main_config, 'score_generation_models') or not main_config.score_generation_models:
        print("ERROR: 'score_generation_models' not found or empty in config.yaml.")
        return

    for model_info_dict in main_config.score_generation_models:
        process_single_model(model_info_dict, main_config)

    print(f"\n{'='*20} Score generation complete {'='*20}")


if __name__ == "__main__":
    main()
