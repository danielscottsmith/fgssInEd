import glob
import pandas as pd
import argparse
from gensim.models.wrappers import LdaMallet
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora import Dictionary
from gensim import corpora
from tqdm import tqdm
import csv
import pickle


parser = argparse.ArgumentParser(
    description="Computes coherence metrics to evaluate mallet model."
)
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="path to interim data",
)
parser.add_argument(
    "-m",
    "--model_dir",
    type=str,
    help="path to models",
)


def get_df(input_dir):
    return pd.read_feather(input_dir + "pro_articles_df.feather")


def get_dictionary(input_dir):
    dictionary = Dictionary.load(input_dir + "dictionary")
    return dictionary


def get_corpus(input_dir):
    corpus = corpora.MmCorpus(input_dir + "corpus.mm")
    return corpus


def get_files(model_dir):
    model_files = glob.glob(model_dir + "*pkl")
    return model_files


def get_coherence(model_files, dictionary, corpus, df, model_dir):
    metrics = ["u_mass", "c_v", "c_npmi", "c_uci"]
    with open(model_dir + "coherence_mallet.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["k", "metric", "coherence"])
        for model_file in tqdm(model_files):
            model = pickle.load(open(model_file, "rb"))
            k = int(model_file.split("mallet")[1].replace("model.pkl", ""))
            for metric in metrics:
                if "c_" in metric:
                    cm = CoherenceModel(
                        model=model,
                        texts=df["tokens"],
                        dictionary=dictionary,
                        coherence=metric,
                    )

                else:
                    cm = CoherenceModel(
                        model=model,
                        corpus=corpus,
                        dictionary=dictionary,
                        coherence=metric,
                    )

                coherence_value = cm.get_coherence()
                row = [k, metric, coherence_value]
                writer.writerow(row)


def main():
    args = parser.parse_args()

    df = get_df(args.input_dir)
    print("\u2713", "Data Loaded!")

    dictionary = get_dictionary(args.input_dir)
    print("\u2713", "Dictionary Loaded!")

    corpus = get_corpus(args.input_dir)
    print("\u2713", "Corpus Loaded!")

    print("Computing coherences...")
    model_files = get_files(args.model_dir)
    get_coherence(model_files, dictionary, corpus, df, args.model_dir)
    print("\u2713", "Coherences computed!")
    print("All done~")


if __name__ == "__main__":
    main()
