#!/usr/bin/env python3
"""
Utility script per preparare labels CSV da struttura ad/cn.
Opzionale: usa questo se vuoi creare un CSV custom con id, text, label.
"""
from __future__ import annotations

import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def create_labels_csv(
    transcripts_root: Path,
    output_csv: Path,
    task: str = "classification"
):
    """
    Crea un CSV con labels derivate dalla struttura delle cartelle ad/cn.
    
    Args:
        transcripts_root: Path to transcripts folder (contains ad/ and cn/)
        output_csv: Output CSV path
        task: "classification" or "regression" (for regression, requires MMSE scores separately)
    """
    data = []
    
    # Processa cartella AD
    ad_dir = transcripts_root / "ad"
    if ad_dir.exists():
        for txt_file in tqdm(sorted(ad_dir.glob("*.txt")), desc="Processing AD samples"):
            sample_id = txt_file.stem
            
            # Leggi trascrizione
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                data.append({
                    'id': sample_id,
                    'text': text,
                    'label': 1 if task == "classification" else None,  # AD = 1
                    'dx': 'ad'
                })
    
    # Processa cartella CN
    cn_dir = transcripts_root / "cn"
    if cn_dir.exists():
        for txt_file in tqdm(sorted(cn_dir.glob("*.txt")), desc="Processing CN samples"):
            sample_id = txt_file.stem
            
            # Leggi trascrizione
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            if text:
                data.append({
                    'id': sample_id,
                    'text': text,
                    'label': 0 if task == "classification" else None,  # CN = 0
                    'dx': 'cn'
                })
    
    # Crea DataFrame
    df = pd.DataFrame(data)
    
    if task == "classification":
        df = df[['id', 'text', 'label', 'dx']]
    else:
        # Per regressione, serve uno script separato che unisce con MMSE scores
        df = df[['id', 'text', 'dx']]
        print("\nNote: For regression task, you need to merge this with MMSE scores separately.")
    
    # Salva CSV
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} samples to {output_csv}")
    print(f"AD samples: {(df['dx'] == 'ad').sum()}")
    print(f"CN samples: {(df['dx'] == 'cn').sum()}")


def merge_with_mmse(
    labels_csv: Path,
    mmse_csv: Path,
    output_csv: Path
):
    """
    Unisce il CSV delle trascrizioni con gli MMSE scores.
    
    Args:
        labels_csv: CSV with id, text, dx
        mmse_csv: CSV with id, mmse scores (e.g., adresso-train-mmse-scores.csv)
        output_csv: Output CSV path
    """
    # Carica i due CSV
    labels_df = pd.read_csv(labels_csv)
    mmse_df = pd.read_csv(mmse_csv)
    
    # Rinomina colonne MMSE CSV per consistenza
    if 'adressfname' in mmse_df.columns:
        mmse_df = mmse_df.rename(columns={'adressfname': 'id'})
    
    # Merge
    merged = labels_df.merge(mmse_df[['id', 'mmse']], on='id', how='left')
    
    # Salva
    merged.to_csv(output_csv, index=False)
    print(f"\nMerged and saved to {output_csv}")
    print(f"Total samples: {len(merged)}")
    print(f"Samples with MMSE: {merged['mmse'].notna().sum()}")


def main():
    parser = argparse.ArgumentParser(description="Prepare labels CSV from ad/cn structure")
    parser.add_argument("--transcripts-root", required=True, help="Path to transcripts folder (contains ad/ and cn/)")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--task", default="classification", choices=["classification", "regression"],
                        help="Task type")
    parser.add_argument("--merge-mmse", help="Path to MMSE scores CSV (for regression task)")
    
    args = parser.parse_args()
    
    transcripts_root = Path(args.transcripts_root)
    output_csv = Path(args.output)
    
    if args.merge_mmse:
        # Prima crea il CSV base
        temp_csv = output_csv.parent / f"temp_{output_csv.name}"
        create_labels_csv(transcripts_root, temp_csv, task=args.task)
        
        # Poi unisci con MMSE
        merge_with_mmse(temp_csv, Path(args.merge_mmse), output_csv)
        
        # Rimuovi temp file
        temp_csv.unlink()
    else:
        # Solo CSV base
        create_labels_csv(transcripts_root, output_csv, task=args.task)


if __name__ == "__main__":
    main()
