#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from scripts import transcribe, train, predict, evaluate, prepare_labels

def main():
    parser = argparse.ArgumentParser(description="Main runner for the ADReSSo21 replication project.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # Transcribe command
    p_transcribe = subparsers.add_parser("transcribe", help="Generate transcripts from audio files.")
    p_transcribe.add_argument("--engine", default="auto", choices=["auto", "nemo", "whisperx", "faster-whisper"], help="ASR backend.")
    p_transcribe.add_argument("--model", default="large-v3", help="ASR model name.")
    p_transcribe.add_argument("--audio-root", required=True, help="Root directory of audio files.")
    p_transcribe.add_argument("--output-root", required=True, help="Output directory for transcripts.")
    p_transcribe.add_argument("--device", default="cuda", help="Device to use.")
    p_transcribe.add_argument("--batch-size", type=int, default=8)
    p_transcribe.add_argument("--num-workers", type=int, default=0)
    p_transcribe.add_argument("--compute-type", default="float16")
    p_transcribe.add_argument("--align", action="store_true")
    p_transcribe.add_argument("--overwrite", action="store_true")
    p_transcribe.set_defaults(func=transcribe.main)

    # Train command
    p_train = subparsers.add_parser("train", help="Train the model with (or without) cross-validation.")
    p_train.add_argument("--config", default="config.yaml", help="Path to the config file.")
    p_train.set_defaults(func=train.main)

    # Predict command
    p_predict = subparsers.add_parser("predict", help="Run prediction and ensembling on the test set.")
    p_predict.add_argument("--config", default="config.yaml", help="Path to the config file.")
    p_predict.add_argument("--test-audio-dir", required=True, help="Directory of test audio files.")
    p_predict.add_argument("--threshold", type=float, default=0.5, help="Classification threshold.")
    p_predict.set_defaults(func=predict.main)
    
    # Evaluate command
    p_evaluate = subparsers.add_parser("evaluate", help="Evaluate predictions against ground truth.")
    p_evaluate.add_argument("--config", default="config.yaml", help="Path to the config file.")
    p_evaluate.add_argument("--predictions", default="predictions_task1.csv", help="Path to predictions CSV.")
    p_evaluate.set_defaults(func=evaluate.main)

    # Prepare labels command
    p_prepare = subparsers.add_parser("prepare-labels", help="Prepare labels CSV from transcript folders.")
    p_prepare.add_argument("--transcripts-root", required=True, help="Path to transcripts folder (contains ad/ and cn/)")
    p_prepare.add_argument("--output", required=True, help="Output CSV path")
    p_prepare.add_argument("--task", default="classification", choices=["classification", "regression"], help="Task type")
    p_prepare.add_argument("--merge-mmse", help="Path to MMSE scores CSV (for regression task)")
    p_prepare.set_defaults(func=lambda args: prepare_labels.main_cli(args))


    # Parse args and call the appropriate function
    args = parser.parse_args()
    
    # Create a new Namespace object that only contains the arguments for the target function
    func_args = argparse.Namespace(**{k: v for k, v in vars(args).items() if k not in ['command', 'func']})
    
    # Call the function with the filtered arguments
    # This is a bit of a hack to make the sub-functions reusable
    # A better approach might be for each main function to accept a list of strings (sys.argv[2:])
    # and do its own parsing, but this works for now.
    if args.command == 'train':
        args.func(args.config)
    elif args.command == 'predict':
        args.func(args.config, args.test_audio_dir, args.threshold)
    elif args.command == 'evaluate':
        args.func() # evaluate.main handles its own parsing
    elif args.command == 'transcribe':
        transcribe.main_cli(args) # transcribe.main needs the full args object
    elif args.command == 'prepare-labels':
        args.func(args)

if __name__ == "__main__":
    # To make imports work correctly, we need to be in the project root.
    # This ensures that `from src...` and `from scripts...` work as expected.
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    main()
