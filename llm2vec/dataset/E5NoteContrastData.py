import json
import random
import os

from .dataset import DataSample, TrainSample, Dataset
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="INFO")

E5_EMBEDDING_PROMPTS = {
    "allnli": [
        "Given a premise, retrieve a hypothesis that is entailed by the premise",
        "Retrieve semantically similar text",
    ],
    "dureader": "Given a Chinese search query, retrieve web passages that answer the question",
    "eli5_question_answer": "Provided a user question, retrieve the highest voted answers on Reddit ELI5 forum",
    "fever": "Given a claim, retrieve documents that support or refute the claim",
    "hotpot_qa": "Given a multi-hop question, retrieve documents that can help answer the question",
    "miracl": "Given a question, retrieve Wikipedia passages that answer the question",
    "mrtydi": "Given a question, retrieve Wikipedia passages that answer the question",
    "msmarco_passage": "Given a web search query, retrieve relevant passages that answer the query",
    "msmarco_document": "Given a web search query, retrieve relevant documents that answer the query",
    "nq": "Given a question, retrieve Wikipedia passages that answer the question",
    "quora_duplicates": [
        "Given a question, retrieve questions that are semantically equivalent to the given question",
        "Find questions that have the same meaning as the input question",
    ],
    "squad": "Retrieve Wikipedia passages that answer the question",
    "t2ranking": "Given a Chinese search query, retrieve web passages that answer the question",
    "trivia_qa": "Retrieve Wikipedia passages that answer the question",
    "notecontrast_data": "Given a patient's medical conditions, retrieve the patient's hospital discharge summary",
}

# Treated completely separate and are mixed into the batches at the end
# Also not shuffled, because already shuffled and some datasets use fixed negatives within the same dataset
MEDICAL_DATASETS = ["notecontrast_data"]

class E5NoteContrastData(Dataset):
    def __init__(
        self,
        dataset_name: str = "E5NoteContrast",
        split: str = "validation",
        file_path: str = "cache/echo-data",
        effective_batch_size: int = 32,
        shuffle_individual_datasets: bool = True,
        separator: str = "!@#$%^&*()",
        # TODO: Adjust based on setting
        ratio_medical: float = 0.05,
        # Original code: 1000 batches of size 512 (8 GPUs with effective batch size 64)
        # Here: Use 4 GPUs and effective batch size 256, so set offset to 2000.
        #       Want to use 1000 batches of size 512, so must use 2000 batches of size 256.
        num_non_medical_batches: int = 2000,
        num_non_medical_batches_offset: int = 2000,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.effective_batch_size = effective_batch_size
        self.shuffle_individual_datasets = shuffle_individual_datasets
        self.separator = separator
        self.ratio_medical = ratio_medical
        self.num_non_medical_batches = num_non_medical_batches
        self.num_non_medical_batches_offset = num_non_medical_batches_offset

        self.data = []
        self.load_data(file_path)

    def __len__(self):
        return len(self.data)
    
    def get_batches(self, data_map):
        # Refactored this method from the original code to reuse it for medical datasets
        datasets = list(data_map.keys())
        all_batches = []
        for dataset in datasets:
            dataset_samples = data_map[dataset]
            for i in range(0, len(dataset_samples), self.effective_batch_size):
                batch = dataset_samples[i : i + self.effective_batch_size]
                if len(batch) == self.effective_batch_size:
                    all_batches.append(batch)
                else:
                    logger.info(f"Skip 1 batch for dataset {dataset}.")
        return all_batches

    def load_data(self, file_path: str = None):
        logger.info(f"Loading E5 data from {file_path}...")

        non_medical_data_map = {}
        all_samples = []
        id_ = 0
        medical_data_map = {}
        for dataset in E5_EMBEDDING_PROMPTS:
            logger.info(f"Loading dataset {dataset}...")
            if dataset in MEDICAL_DATASETS:
                if dataset not in medical_data_map:
                    medical_data_map[dataset] = []
            else:
                if dataset not in non_medical_data_map:
                    non_medical_data_map[dataset] = []
            with open(os.path.join(file_path, f"{dataset}.jsonl"), "r") as f:
                dataset_samples = f.readlines()

            dataset_samples = [json.loads(d) for d in dataset_samples]

            for i, sample in enumerate(dataset_samples):
                instruction = (
                    E5_EMBEDDING_PROMPTS[dataset]
                    if isinstance(E5_EMBEDDING_PROMPTS[dataset], str)
                    else E5_EMBEDDING_PROMPTS[dataset][i % 2]
                )
                query = f"{instruction}; " + self.separator + sample["query"]
                if dataset in [
                    "allnli_split2",
                    "quora_duplicates_split1",
                    "quora_duplicates_split2",
                ]:
                    pos = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + sample["positive"]
                    )
                    neg = (
                        f"{E5_EMBEDDING_PROMPTS[dataset]}; "
                        + self.separator
                        + sample["negative"]
                    )
                else:
                    pos = self.separator + sample["positive"]
                    neg = self.separator + sample["negative"]

                if dataset in MEDICAL_DATASETS:
                    medical_data_map[dataset].append(id_)
                else:
                    non_medical_data_map[dataset].append(id_)
                
                all_samples.append(
                    DataSample(
                        id_=id_,
                        query=query,
                        positive=pos,
                        negative=neg,
                        task_name=dataset,
                    )
                )
                id_ += 1

        # Combine split1 and split2
        new_data_map = {}
        for dataset in non_medical_data_map:
            new_dataset = dataset.replace("_split1", "").replace("_split2", "")
            if new_dataset not in new_data_map:
                new_data_map[new_dataset] = []
            new_data_map[new_dataset] += non_medical_data_map[dataset]
        non_medical_data_map = new_data_map

        if self.shuffle_individual_datasets:
            for task, samples in non_medical_data_map.items():
                random.shuffle(samples)

        logger.info(
            f"Batching Echo data properly for effective batch size of {self.effective_batch_size}..."
        )
        
        all_batches = self.get_batches(non_medical_data_map)
        random.shuffle(all_batches)
        
        # In original code, the first sample now has id 1302898 and there are 23582 batches
        # TODO: Careful sanity check for batch sizes 64 and 512 based on original code
        assert ((self.effective_batch_size != 64) or (all_batches[0][0] == 1302898 and len(all_batches) == 23582)) and \
            ((self.effective_batch_size != 512) or (all_batches[0][0] == 1382680 and len(all_batches) == 2942))
        
        # Now subselect number of batches and replace random batches with medical batches of medical datasets
        logger.info(f"Selecting {self.num_non_medical_batches} (w/ offset {self.num_non_medical_batches_offset}) of {len(all_batches)} non-medical batches.")
        all_batches = all_batches[self.num_non_medical_batches_offset:self.num_non_medical_batches_offset + self.num_non_medical_batches]
        # Combination of splits not needed for medical datasets
        # Prepare batches from medical datasets
        if self.ratio_medical > 0:
            total_batches = len(all_batches)
            num_medical_batches = int(total_batches * self.ratio_medical)
            medical_batches = self.get_batches(medical_data_map)
            num_unique_medical_batches = len(medical_batches)
            random.shuffle(medical_batches)
            medical_batches_idx = [i % len(medical_batches) for i in range(num_medical_batches)]
            medical_batches = [medical_batches[i] for i in medical_batches_idx]
            logger.info(f"Selected {num_medical_batches} medical batches from {num_unique_medical_batches} unqiue batches.")
        else:
            logger.info("No medical batches selected.")
            medical_batches = []
            
        num_medical_batches = len(medical_batches)
        assert num_medical_batches <= len(all_batches), f"Selected {num_medical_batches} medical batches but only {len(all_batches)} batches available to replace."
        
        # Replace some original batches with medical batches
        replace_idx = list(range(len(all_batches)))
        random.shuffle(replace_idx)
        replace_idx = replace_idx[:num_medical_batches]
        for i, idx in enumerate(replace_idx):
            all_batches[idx] = medical_batches[i]
        
        # Now combine all samples via their indices
        final_idx_order = []
        for batch in all_batches:
            for idx in batch:
                final_idx_order.append(idx)

        self.data = [all_samples[idx] for idx in final_idx_order]
        
        # Sanity check: ensure that batches in self.data are of same task with correct query
        task_name = ""
        queries = []
        for i, sample in enumerate(self.data):
            if i % self.effective_batch_size == 0:
                task_name = sample.task_name
                queries = E5_EMBEDDING_PROMPTS[task_name] if isinstance(E5_EMBEDDING_PROMPTS[task_name], list) else [E5_EMBEDDING_PROMPTS[task_name]]
            assert sample.task_name == task_name, f"Task name mismatch at index {i}."
            assert any([sample.query.startswith(query) for query in queries]), f"Query mismatch at index {i}."

        logger.info(f"Loaded {len(self.data)} samples (MM).")

    def __getitem__(self, index):
        sample = self.data[index]
        if self.split == "train":
            return TrainSample(
                texts=[sample.query, sample.positive, sample.negative], label=1.0
            )
        elif self.split == "validation":
            assert False, "E5Data does not have a validation split."
