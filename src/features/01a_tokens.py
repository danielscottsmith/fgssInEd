import pandas as pd
import argparse
from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from gensim.models import Phrases
from gensim.models.phrases import Phraser


parser = argparse.ArgumentParser(
    description="Stem and then writes dictionary obj and serilized corpus obj"
)
parser.add_argument(
    "-i",
    "--in_dir",
    type=str,
    help="data subdirectory",
)


def get_df(input_dir):
    df = pd.read_feather(input_dir + "cleaned.feather")
    return df


def preprocess_text(df, input_dir):
    tqdm.pandas()
    df["tokens"] = df["text"].progress_apply(lambda x: preprocess_string(" ".join(x)))
    df.to_feather(input_dir + "cleaned.feather")
    return df


def get_bigrams(df, input_dir):
    bigrams = Phrases(df["tokens"], min_count=50)
    bigrams = Phraser(bigrams)
    for idx in range(len(df["tokens"])):
        for tkn in bigrams[df["tokens"].iloc[idx]]:
            if "_" in tkn:
                df["tokens"].iloc[idx].append(tkn)
    df.to_feather(input_dir + "cleaned.feather")
    return df


def make_dictionary(df, input_dir):
    dictionary = Dictionary(df["tokens"])
    dictionary.filter_extremes(no_above=99)
    dictionary.filter_extremes(no_below=1)
    dictionary.save(input_dir + "dictionary")
    return dictionary


def make_corpus(dictionary, df, input_dir):
    corpus = [dictionary.doc2bow(doc) for doc in df["tokens"]]
    corpora.MmCorpus.serialize(input_dir + "corpus.mm", corpus)


def main():
    args = parser.parse_args()

    print("Loading data...")
    df = get_df(args.in_dir)
    print("\u2713", "Data loaded!")

    print("Tokenizing and stemming data...")
    df = preprocess_text(df, args.in_dir)
    print(u'\u2713', "Data preprocessed!")

    print("Identifying bigrams...")
    df = get_bigrams(df, args.in_dir)
    print(u'\u2713', "Bigrams identified!")

    dictionary = make_dictionary(df, args.in_dir)
    print("\u2713", "Dictionary written!")

    make_corpus(dictionary, df, args.in_dir)
    print("\u2713", "Corpus written!")
    print("All Done~")


if __name__ == "__main__":
    main()
