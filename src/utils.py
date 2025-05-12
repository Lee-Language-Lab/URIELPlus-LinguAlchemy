import torch
from transformers import AutoConfig
from datasets import load_dataset, DatasetDict, concatenate_datasets, load_from_disk
from tqdm import tqdm
import random
import numpy as np

import os

CUR_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHED_DATASET_DIR = os.path.join(CUR_DIR, "cached_dataset")

DATASET_NAME_TO_HF_DATASET_MAPPING = {
    "massive": "AmazonScience/massive",
    "masakhanews": "masakhane/masakhanews",
    "semrel": "SemRel/SemRel2024",
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    

# Simple utility function which allows downloading data for offline machine
# An interface to get proper dataset with the choice of caching or no cache
def get_dataset(dataset_name, lang, split=None, cache=True):
    if split:
        cached_dataset_directory = os.path.join(CACHED_DATASET_DIR, f"{dataset_name}/{lang}/{split}")
    else:
        cached_dataset_directory = os.path.join(CACHED_DATASET_DIR, f"{dataset_name}/{lang}")

    # If cache, check first if it exists. If it does, simply load from cached directory
    if cache and os.path.isdir(cached_dataset_directory):
        return load_from_disk(cached_dataset_directory)
    
    if split:
        data = load_dataset(DATASET_NAME_TO_HF_DATASET_MAPPING[dataset_name], lang, split=split)
    else:
        data = load_dataset(DATASET_NAME_TO_HF_DATASET_MAPPING[dataset_name], lang)
    
    # If cache, save to disk
    if cache:
        data.save_to_disk(cached_dataset_directory)
    return data

def get_massive_dataset(model_name, tokenizer, uriel_vector, lang_to_index, debug=False):
    # Language Codes
    complete_langs = [
        "af-ZA", "am-ET", "ar-SA", "az-AZ", "bn-BD", "ca-ES", "cy-GB", "da-DK", "de-DE",
        "el-GR", "en-US", "es-ES", "fa-IR", "fi-FI", "fr-FR", "he-IL", "hi-IN", "hu-HU",
        "hy-AM", "id-ID", "is-IS", "it-IT", "ja-JP", "jv-ID", "ka-GE", "km-KH", "kn-IN",
        "ko-KR", "lv-LV", "ml-IN", "mn-MN", "ms-MY", "my-MM", "nb-NO", "nl-NL", "pl-PL",
        "pt-PT", "ro-RO", "ru-RU", "sl-SL", "sq-AL", "sv-SE", "sw-KE", "ta-IN", "te-IN",
        "th-TH", "tl-PH", "tr-TR", "ur-PK", "vi-VN", "zh-CN", "zh-TW"
    ]

    train_langs = [
        "ar-SA", "hy-AM", "bn-BD", "my-MM", "zh-CN", "zh-TW", "en-US", "fi-FI", "fr-FR",
        "ka-GE", "de-DE", "el-GR", "hi-IN", "hu-HU", "is-IS", "id-ID", "ja-JP", "jv-ID",
        "ko-KR", "lv-LV", "pt-PT", "ru-RU", "es-ES", "vi-VN", "tr-TR",
    ]

    # Loading and processing datasets
    dataset_train, dataset_valid = [], []
    for lang in tqdm(train_langs):
        dataset_train.append(
            get_dataset("massive", lang, "train").remove_columns(
                ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
            )
        )
        dataset_valid.append(
            get_dataset("massive", lang, "validation").remove_columns(
                ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
            )
        )

    dset_dict = DatasetDict({
        "train": concatenate_datasets(dataset_train),
        "valid": concatenate_datasets(dataset_valid),
    })

    dataset_test = {}
    for lang in tqdm(complete_langs):
        dataset_test[lang] = get_dataset("massive", lang, "test").remove_columns(
            ["id", "partition", "scenario", "annot_utt", "worker_id", "slot_method", "judgments"]
        )

    dset_test_dict = DatasetDict(dataset_test)

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        lang_label = batch['locale'][0]
        encoding = tokenizer(batch["utt"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        lang_index = lang_to_index[lang_label]
        encoding['language_labels'] = torch.tensor([lang_index] * len(batch['utt']))
        uriel_vec = uriel_vector[lang_index]
        encoding['uriel_labels'] = uriel_vec.repeat(len(batch['utt']), 1)
        
        return encoding

    if "intent" not in dset_dict.column_names:
        dset_dict = dset_dict.rename_column("intent", "labels")
    if "intent" not in dset_test_dict.column_names:
        dset_test_dict = dset_test_dict.rename_column("intent", "labels")

    dset_dict = dset_dict.map(encode_batch, batched=True)
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True)

    # Initialize model
    config = AutoConfig.from_pretrained(model_name, num_labels=60)
    if model_name == "bert-base-multilingual-cased":
        dset_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "token_type_ids", "attention_mask", "language_labels", "uriel_labels"])
        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "token_type_ids", "attention_mask", "language_labels", "uriel_labels"])
    elif model_name == "xlm-roberta-base":
        dset_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "attention_mask", "language_labels", "uriel_labels"])
        dset_test_dict.set_format(type="torch", columns=["labels", "utt", "input_ids", "attention_mask", "language_labels", "uriel_labels"])
        
    return config, dset_dict, dset_test_dict
        
def get_masakhanews_dataset(model_name, tokenizer, uriel_vector, lang_to_index, debug=False):
    train_langs = ["amh","eng", "fra", "hau", "swa", "orm", "som"]
    test_langs = ["ibo", "lin", "lug", "pcm", "run", "sna", "tir", "xho", "yor"]

    dataset_train, dataset_valid, dataset_test = [], [], {}
    columns_to_remove = ["text", "headline_text", "url"]

    for lang in train_langs:
        dataset = get_dataset("masakhanews", lang)
        dataset_train.append(dataset['train'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False))
        dataset_valid.append(dataset['validation'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False))

    for lang in train_langs + test_langs:
        dataset = get_dataset("masakhanews", lang)
        dataset_test[lang] = dataset['test'].remove_columns(columns_to_remove).map(lambda example: {'lang': lang}, batched=False)

    dset_dict = DatasetDict({
        'train': concatenate_datasets(dataset_train),
        'valid': concatenate_datasets(dataset_valid)
    })
    dset_test_dict = DatasetDict(dataset_test)

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        lang_label = batch['lang'][0]  # Use the first locale in the list as the language label
        encoding = tokenizer(batch["headline"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        # Language labels as indices
        lang_index = lang_to_index[lang_label]

        # URIEL vectors
        uriel_vec = uriel_vector[lang_index]
        
        encoding['uriel_labels'] = uriel_vec.repeat(len(batch['headline']), 1)
        return encoding

    dset_dict = dset_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])

    config = AutoConfig.from_pretrained(model_name, num_labels=7)

    return config, dset_dict, dset_test_dict

def get_semrel_dataset(model_name, tokenizer, uriel_vector, lang_to_index, debug=False):
    train_langs = ["amh", "arq", "ary", "eng", "esp", "hau", "kin", "mar", "tel"]
    test_langs = ["afr", "arb", "hin", "ind", "pan"]
    
    dataset_train, dataset_valid, dataset_test = [], [], {}

    for lang in train_langs:
        dataset = get_dataset("semrel", lang)
        dataset_train.append(dataset['train'].map(lambda example: {'lang': lang}, batched=False))
        dataset_valid.append(dataset['dev'].map(lambda example: {'lang': lang}, batched=False))

    for lang in train_langs + test_langs:
        dataset = get_dataset("semrel", lang)
        dataset_test[lang] = dataset['test'].map(lambda example: {'lang': lang}, batched=False)

    dset_dict = DatasetDict({
        'train': concatenate_datasets(dataset_train),
        'valid': concatenate_datasets(dataset_valid)
    })
    dset_test_dict = DatasetDict(dataset_test)

    def encode_batch(batch):
        """Encodes a batch of input data using the model tokenizer."""
        lang_label = batch['lang'][0]  # Use the first locale in the list as the language label
        encoding = tokenizer(batch["sentence1"], max_length=80, truncation=True, padding="max_length", return_tensors="pt")
        
        # Language labels as indices
        if lang_label == "esp":
            lang_index = lang_to_index["spa"]
        else:
            lang_index = lang_to_index[lang_label]

        # URIEL vectors
        uriel_vec = uriel_vector[lang_index]
        
        encoding['uriel_labels'] = uriel_vec.repeat(len(batch['sentence1']), 1)
        return encoding
    
    config = AutoConfig.from_pretrained(model_name, num_labels=7)

    dset_dict = dset_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])
    dset_test_dict = dset_test_dict.map(encode_batch, batched=True).rename_column("label", "labels").with_format(type="torch", columns=["input_ids", "attention_mask", "labels", "uriel_labels"])

    return config, dset_dict, dset_test_dict
