##### CODE ADAPTED from LINGUALCHEMY REPOSITORY https://github.com/faridlazuarda/LinguAlchemy/blob/main/src/lingualchemy.py #####

import argparse
import torch
import shutil
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from scipy import stats
from tqdm import tqdm
import os
import json

from src.fusion_models import *
from src.utils import *

def parse_scale(value):
    try:
        return int(value)
    except ValueError:
        if value == "dynamiclearn" or value == "dynamicscale":
            return value
        else:
            raise ValueError(f"Scale should be an integer, 'dynamiclearn', or 'dynamicscale' but not {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Set the pre-trained model.",
    )  # "bigscience/mt0-base"
    parser.add_argument(
        "--epochs",
        type=int,
        default=30, # Adjusted based on the paper
        help="The epochs set for training.",
    )
    parser.add_argument(
        "--scale",
        type=parse_scale,
        default="10",
        help="The uriel scale set for training.",
    )
    parser.add_argument(
        "--vector",
        type=str,
        default=10,
        help="The uriel vector set for training.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="massive",
        help="The dataset for training and test.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Only train on the first 100 examples.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.path.join(CUR_DIR, "ablation"),
        help="Set the pre-trained model.",
    )
    parser.add_argument(
        "--eval_path",
        type=str,
        default=os.path.join(CUR_DIR, "outputs"),
        help="Set the evaluation dump file path.",
    )
    parser.add_argument(
        "--override_results",
        action="store_true",
        default=False,
        help="When enabled, remove the previous checkpoints results.",
    )
    parser.add_argument(
        "--wandb_offline", default=False, action="store_true", help="wandb offline mode"
    )
    
    args = parser.parse_args()

    if os.path.exists(args.out_path):
        assert (
            args.debug or args.override_results
        ), f"Output dir {args.out_path} already exists!"
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path)

    os.environ['TOKENIZERS_PARALLELISM'] = 'true'

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["WANDB_DISABLED"] = "true"
    
    set_seed(RANDOM_SEED)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load URIEL data
    uriel_data = torch.load(os.path.join(CUR_DIR, f"vectors/{args.dataset}/{args.vector}.pt"))
    uriel_vector = torch.stack([torch.tensor(uriel_data[lang]) for lang in sorted(uriel_data.keys())])
    lang_to_index = {lang: idx for idx, lang in enumerate(sorted(uriel_data.keys()))}
    
    # Select dataset based on argument
    if args.dataset == "massive":
        config, dset_dict, dset_test_dict = get_massive_dataset(args.model_name, tokenizer, uriel_vector, lang_to_index, args.debug)
    elif args.dataset == "masakhanews":
        config, dset_dict, dset_test_dict = get_masakhanews_dataset(args.model_name, tokenizer, uriel_vector, lang_to_index, args.debug)
    elif args.dataset == "semrel":
        config, dset_dict, dset_test_dict = get_semrel_dataset(args.model_name, tokenizer, uriel_vector, lang_to_index, args.debug)

    # Set model
    if args.model_name == "bert-base-multilingual-cased":
        model = FusionBertForSequenceClassification(config, uriel_vector)
    elif args.model_name == "xlm-roberta-base":
        model = FusionXLMRForSequenceClassification(config, uriel_vector)
    else:
        raise ValueError(f"Model name {args.model_name} is not recognizable. Please only provide either 'bert-base-multilingual-cased' or 'xlm-roberta-base'")

    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    training_args = TrainingArguments(
        output_dir=args.out_path,
        save_strategy="epoch",
        save_total_limit=2,
        learning_rate=5e-5,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=100,
        dataloader_num_workers=16,
        seed=RANDOM_SEED,
    )
    
    # Select the corresponding trainer based on scale
    if args.scale == "dynamicscale":
        trainer = CustomTrainerDynamicscale(
            model=model,
            args=training_args,
            train_dataset=dset_dict['train'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingEpochCallback(early_stopping_patience=3)],
            lang_vec=uriel_vector,
        )
    elif args.scale == "dynamiclearn":
        trainer = CustomTrainerDynamiclearn(
            model=model,
            args=training_args,
            train_dataset=dset_dict['train'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingEpochCallback(early_stopping_patience=3)],
            lang_vec=uriel_vector,
        )
    else:
        # This should be an integer based on parsing
        trainer = CustomTrainer(
            model=model,
            config=config,
            args=training_args,
            train_dataset=dset_dict['train'],
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingEpochCallback(early_stopping_patience=3)],
            lang_vec=uriel_vector,
            scale=int(args.scale)
        )

    trainer.train()

    trainer.model.save_pretrained(args.out_path)
    tokenizer.save_pretrained(args.out_path)

    model.eval()
    results = {}

    all_preds, all_labels = [], []

    for lang, dataset in dset_test_dict.items():
        test_loader = DataLoader(dataset, batch_size=64)
        for batch in tqdm(test_loader, desc=f"Testing lang {lang}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            with torch.no_grad():
                logits, lang_logits, pooled_output = model(input_ids, attention_mask, None, None, None)[0]
                preds = torch.argmax(logits, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        if args.dataset == "massive" or args.dataset == "masakhanews":
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(all_labels, all_preds, average='micro')
            accuracy = accuracy_score(all_labels, all_preds)

            results[lang] = {
                'precision_macro': precision_macro,
                'precision_micro': precision_micro,
                'recall_macro': recall_macro,
                'recall_micro': recall_micro,
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'accuracy': accuracy
            }
        else:
            # Calculate pearson correlation instead
            results[lang] = {
                "pearson": stats.pearsonr(all_labels, all_preds)
            }

    results_file_path = f"{args.eval_path}/{args.vector}_scores.json"
    if os.path.exists(results_file_path):
        if args.override_results:
            os.remove(results_file_path)
        else:
            raise AssertionError(f"Output file {results_file_path} already exists!")

    os.makedirs(f"{args.eval_path}", exist_ok=True)

    with open(results_file_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print(results)
