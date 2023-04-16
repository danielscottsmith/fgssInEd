import pandas as pd
import argparse
from tqdm import tqdm
from gensim.parsing.preprocessing import preprocess_string
from gensim.corpora import Dictionary

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "-i",
    "--in_dir",
    type=str,
    help="interim data subdirectory",
)


EMP = [
    # "data analysis",
    "data collection",
    # "data methods",
    # "empirical analysis",
]

REG = [
    "regression model",
    # "regression analysis",
    # "statistical model",
    # "hypothesis test" ,
    # "regression models",
    # "statistical models",
    # "descriptive statistics",
]


def get_df(input_dir):
    df = pd.read_feather(input_dir + "cleaned.feather")
    # df = pd.read_csv(input_dir + "cleaned.csv")
    return df


def get_dictionary(input_dir):
    dictionary = Dictionary.load(input_dir + "dictionary")
    return dictionary


def _code_article(tokens):
# def _code_article(tokens, queries):
    # freq = 0
    # for query in queries:
        # query = "_".join(preprocess_string(query))
        # freq += list(tokens).count(query)
    freq = list(tokens).count('data')
    return freq


def code_articles(df, input_dir):
    tqdm.pandas()
    # df["emp"] = df["tokens"].progress_apply(lambda tokens: _code_article(tokens, EMP))
    df["emp"] = df["tokens"].progress_apply(lambda tokens: _code_article(tokens))
    print(df['emp'].describe())
    df.to_feather(input_dir + "cleaned.feather")


def main():
    args = parser.parse_args()
    print("Loading data...")
    
    df = get_df(args.in_dir)
    print("\u2713", "Data loaded!")
    
    
    # dictionary = get_dictionary(args.in_dir)
    # print(dictionary.doc2bow[df['tokens'].iloc[10]])

    print("Encoding 'empirical' & 'regression' methods...")
    code_articles(df, args.in_dir)
    print("\u2713", "Methods encoded!")

    print("All Done~")


if __name__ == "__main__":
    main()
