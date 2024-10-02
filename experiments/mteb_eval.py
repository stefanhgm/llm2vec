import argparse
import mteb
from mteb.benchmarks import MTEB_MAIN_EN
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    )
    parser.add_argument("--task_name", type=str, default="STS16")
    parser.add_argument("--task_types", type=str, default="")
    parser.add_argument("--do_mteb_main_en", action="store_true", default=False)
    parser.add_argument(
        "--task_to_instructions_fp",
        type=str,
        default="test_configs/mteb/task_to_instructions.json",
    )
    parser.add_argument("--output_dir", type=str, default="results")

    args = parser.parse_args()

    model_kwargs = {}
    if args.task_to_instructions_fp is not None:
        with open(args.task_to_instructions_fp, "r") as f:
            task_to_instructions = json.load(f)
        model_kwargs["task_to_instructions"] = task_to_instructions

    model = mteb.get_model(args.model_name, **model_kwargs)

    if args.do_mteb_main_en:
        tasks_orig = mteb.get_tasks(tasks=MTEB_MAIN_EN.tasks, languages=["eng"])
        # Remove MSMARCO
        # "Note that the public leaderboard uses the test splits for all datasets except MSMARCO, where the "dev" split is used."
        # See: https://github.com/embeddings-benchmark/mteb
        tasks = [t for t in tasks_orig if "MSMARCO" not in t.metadata.name]
        assert len(tasks_orig) == 67 and len(tasks) == 66
    elif args.task_types:
        tasks = mteb.get_tasks(task_types=[args.task_types], languages=["eng"])
    else:
        tasks = mteb.get_tasks(tasks=[args.task_name], languages=["eng"])

    evaluation = mteb.MTEB(tasks=tasks)

    # Set logging to debug
    mteb.logger.setLevel(mteb.logging.DEBUG)
    # Only run on test set for leaderboard, exception: MSMARCO manually on dev set
    results = evaluation.run(model, eval_splits=["test"], verbosity=2, output_folder=args.output_dir)
