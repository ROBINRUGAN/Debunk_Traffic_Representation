import os
import hashlib
import pickle
import torch
import pandas as pd
from torch.utils.data import Dataset, RandomSampler
from tqdm import tqdm

from sklearn.model_selection import train_test_split


class Classification_Dataset(Dataset):
    def __init__(self, opts, tokenizer):
        self.tokenizer = tokenizer
        self.q_len = opts["max_qst_length"]
        self.t_len = opts["max_ans_length"]
        self.seed = opts["seed"]
        self.batch = opts["batch_size"]
        # In validation, loss is not needed
        if "loss" in opts.keys():
            self.type_loss = opts["loss"]

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        question = self.questions
        context = self.context[idx]
        answer = self.answer[idx]
        # parquet/多卡 可能得到 numpy.str_ 或 Series，tokenizer 要求 Python str
        if isinstance(question, pd.Series):
            question = question.iloc[0] if len(question) else ""
        if isinstance(context, pd.Series):
            context = context.iloc[0] if len(context) else ""
        question = str(question) if pd.notna(question) else ""
        context = str(context) if pd.notna(context) else ""
        question_tokenized = self.tokenizer.tokenize_question(question, context)
        return idx, {
            "input_ids": torch.tensor(
                question_tokenized["input_ids"], dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                question_tokenized["attention_mask"], dtype=torch.long
            ),
            "label_class": torch.tensor(answer)
        }

    def _process_context(self, ser, format_input, desc="context", show_pbar=True):
        disable_pbar = show_pbar and (os.environ.get("RANK", "0") != "0")
        out = [pkt.replace(" ", "") for pkt in tqdm(ser, desc=desc, disable=disable_pbar)]
        if format_input == "every4":
            out = [
                "".join([str(pkt[i : i + 4]) + " " for i in range(0, len(pkt), 4)]).strip()
                for pkt in tqdm(out, desc=f"{desc} every4", disable=disable_pbar)
            ]
        elif format_input == "every2":
            out = [
                "".join([str(pkt[i : i + 2]) + " " for i in range(0, len(pkt), 2)]).strip()
                for pkt in tqdm(out, desc=f"{desc} every2", disable=disable_pbar)
            ]
        return out

    def _cache_dir_and_key(self, type, input_path, format_input, input_validation, percentage):
        """缓存目录与 key：处理一次后存 pickle，下次同参数直接反序列化，类型与内存中完全一致。"""
        base = os.path.dirname(os.path.abspath(input_path))
        cache_dir = os.path.join(base, ".processed_cache")
        raw = f"{input_path}|{input_validation}|{format_input}|{percentage}|{self.seed}|{self.batch}|{type}"
        key = hashlib.md5(raw.encode()).hexdigest()[:16]
        return cache_dir, key

    def _load_from_cache(self, cache_dir, key):
        """若存在 pickle 缓存则加载并返回 True，否则返回 False。二进制还原，无类型歧义。"""
        path = os.path.join(cache_dir, f"{key}.pkl")
        if not os.path.isfile(path):
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.train_data = state["train_data"]
        self.val_data = state["val_data"]
        self.data = state["data"]
        self.size_train = state["size_train"]
        self.size_val = state.get("size_val")
        self.questions = self.data["question"].iloc[0]
        self.context = self.data["context"]
        self.answer = self.data["class"]
        if hasattr(self, "type_loss"):
            self.retrieveCardinality()
        return True

    def _save_to_cache(self, cache_dir, key):
        """把当前 train_data / val_data / data 等完整状态用 pickle 存成二进制，原样还原。"""
        os.makedirs(cache_dir, exist_ok=True)
        state = {
            "train_data": self.train_data,
            "val_data": getattr(self, "val_data", None),
            "data": self.data,
            "size_train": self.size_train,
            "size_val": getattr(self, "size_val", None),
        }
        path = os.path.join(cache_dir, f"{key}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_dataset(self, type, input_path, format_input, input_validation="", percentage=100):
        """
        load_dataset
        ------------
        The method is used to load the data in the class.
        若存在与参数对应的已处理缓存（.processed_cache），则直接加载；否则读 parquet、处理、再写入缓存。
        Args:
            - input_path: must be a csv file were a pandas dataframe is stored
        """
        cache_dir, key = self._cache_dir_and_key(type, input_path, format_input, input_validation, percentage)
        if self._load_from_cache(cache_dir, key):
            if os.environ.get("RANK", "0") == "0":
                print(f"Loaded processed dataset from cache ({cache_dir}, key={key}).", flush=True)
            return

        data_desc = "test" if type == "Test" else "train"
        if os.environ.get("RANK", "0") == "0":
            print(f"Reading {data_desc}.parquet (may take 1–2 min for large files)...", flush=True)
        self.train_data = pd.read_parquet(input_path)
        self.train_data["context"] = self._process_context(
            self.train_data.context, format_input, desc=data_desc
        )
        num_rows_train = int(len(self.train_data) * percentage / 100)
        self.train_data = self.train_data.sample(n=num_rows_train, random_state=self.seed)
        multiple_batch = int(len(self.train_data) / self.batch)
        self.train_data = self.train_data[: multiple_batch * self.batch]
        self.size_train = len(self.train_data)
        self.train_data = self.train_data.reset_index()

        if input_validation != "":
            if os.environ.get("RANK", "0") == "0":
                print("Reading val.parquet...", flush=True)
            self.val_data = pd.read_parquet(input_validation)
            self.val_data["context"] = self._process_context(
                self.val_data.context, format_input, desc="val"
            )
            self.val_data.index += self.size_train
            num_rows_val = int(len(self.val_data) * percentage / 100)
            self.val_data = self.val_data.sample(n=num_rows_val, random_state=self.seed)
            multiple_batch = int(len(self.val_data) / self.batch)
            self.val_data = self.val_data[: multiple_batch * self.batch]
            self.size_val = len(self.val_data)

            self.data = pd.concat([self.train_data, self.val_data], axis=0)
        elif type == "Train":
            self.data = self.train_data
            self.train_data, self.val_data = train_test_split(
                self.data, test_size=0.2, random_state=self.seed
            )
        else:
            self.data = self.train_data

        self.questions = self.data["question"].iloc[0]
        self.context = self.data["context"]
        self.answer = self.data["class"]
        if hasattr(self, "type_loss"):
            self.retrieveCardinality()

        if os.environ.get("RANK", "0") == "0":
            self._save_to_cache(cache_dir, key)
            print(f"Saved processed dataset to cache ({cache_dir}, key={key}).", flush=True)

    def create_test_sampler(self):
        self.test_sampler = RandomSampler(self.data.index)

    def retrieveTypes(self):
        elems = self.data["type_q"].unique()
        return elems.tolist()
    
    def retrieveCardinality(self):
        elems = self.data["class"].value_counts()
        if self.type_loss == "weighted":
            total_samples = sum(elems)
            class_weights = [pow(1 - count / total_samples, 2) for count in elems]
            self.weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.weights = torch.tensor([1 for i in range(len(elems))], dtype=torch.float32)
    
    def create_trainVal_sampler(self):
        self.train_sampler = RandomSampler(self.train_data.index)
        self.val_sampler = RandomSampler(self.val_data.index)

    # GETTERS
    def get_weights(self):
        return self.weights

    def get_train_sampler(self):
        return self.train_sampler

    def get_val_sampler(self):
        return self.val_sampler

    def get_test_sampler(self):
        return self.test_sampler

    def get_class_Byindex(self, index):
        return self.type_q.iloc[index]

    def get_if_time(self):
        return False
    
    def get_classification_stats(self):
        tmp = self.val_data.sort_values('class')
        labels = list(tmp["type_q"].drop_duplicates())
        return len(labels), labels

    def get_classification_test_stats(self):
        tmp = self.data.sort_values('class')
        labels = list(tmp["type_q"].drop_duplicates())
        return len(labels), labels
