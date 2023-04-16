# Replication Package
>### Women, World Society, and the Expansion of Academic Knowledge in Education.

This is a working project directory and will be made into a stable reproduction package, _eventually_. The main goal is to understand the evolution of research on gender within the field of education using every article published in journals with "educ*" or "teach" in the title and hosted by JSTOR, 1900–2022. 

Currently, the strategy is to run topic models to identify the latent topic of gender/sex/sexuality, and then explain variation in article's focus on that topic over time using logitudinal regression models.


## Project Organization

Below is a map of the directory with annotations for general orientation.

    ├── LICENSE
    ├── README.md            <- The top-level README for researchers using this project.
    │    
    ├── models                <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks             <- Jupyter notebooks. Naming convention is a number (for order of analysis).
    │
    ├── references            <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports               <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures           <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt      <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py              <- makes project pip installable (pip install -e .) so src can be imported
    └── src                   <- Source code for use in this project.
        ├── __init__.py       <- Makes src a Python module
        │
        ├── data              <- Scripts to prep data.
        │   ├── 00_clean_jstor.py
        │   └── 00_clean_cer.py
        │
        ├── features          <- Script to turn interim data into bag of words, dictionary, and corpus objects.
        │   ├── 01_infer_genders.py
        │   └── 01_preprocess_texts.py
        │
        ├── models            <- Scripts to train & evaluate MALLET models.
        │   ├── predict_model.py
        │   ├── eval_mallet.py
        │   └── train_mallet.py
        │
        └── visualization  <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py

## Requesting JSTOR data.

The following `data` subdirectory is required to fully reproduce this project. The full JSTOR data is under a single-use license and must be attained directly from Constellate/ITHAKA for replication, in compliance with paragraph 4 of Constellate.org's [Terms and Conditions](constellate.org/terms-and-conditions) (accessed Dec. 2022). A publicly available data abstract including query details, meta data, and ngrams of these data can nontheless be downloaded and explored at [Constellate.org](https://constellate.org/dataset/dcaf743a-a4e1-39cb-5ac6-024c6b5d9c53/). 

    ├── data                 <- All data
        ├── JSTOR            <- Population of articles published in journals with "educ*" or "teach*" in titles on JSTOR
        │   ├── interim      <- Intermediate data that has been transformed.
        │   ├── processed    <- The final, canonical data sets for modeling.
        │   └── raw          <- The original, immutable data dump from JSTOR.
        │
        └── CER               <- ca. 1200, hand-labeled articles from Comparative Education Review 


## Cleaning data 
### JSTOR
This script should be used to decompress and extract relevant data from the original JSTOR data dump.

Example commands: 

```
python src/data/00_clean_jstor.py -i data/JSTOR/raw/ -o data/JSTOR/interim/ 
```

### Cleaning CER data 

Example commands: 

```
python src/01_clean_cer.py -i -o 
```


## Get features
All features take as input `cleaned.feather` and write as output `cleaned.feather`. This means the order of executing scripts doesn't matter; features are subsequently added and read in but not overwritten. So, if `01_infer_genders.py` is executed first, this adds the gender to `cleaned.feather` which then can be read by `01_preprocess_texts.py`. Vice versa holds. 

### Preprocess Texts
This script tokenizes all articles and produces serialized gensim corpus and dictionary objects. These can take a lot of time! The feature added is named `tokens`.

### Infer Genders
This script infers the conventional gender signaled by the first author's pen name using genderize.io. This can take a lot of time! And it requires a subscription. For convenience, I include a csv of first name's and their gender in the `data/JSTOR/interm` folder. If there exists a csv in that directory, the script first checks to see if the name of is already genderized; if not, only then does it look it up. This is good for iteration and adding more / novel data.

### Classify methods
These scripts classify whether an article uses quantitative methods. First it trains on the labeled CER data. Then it evaluates. Finally, it predicts on the rest of the sample. This can take a lot of time! 


## Train mallet 
This script uses the 131K random sample of JSTOR education articles to find the model of size k that has the modal maximum value across four coherence metrics. This can take a lot of time! 

## Predict mallet
This script loads the best fit model and then predicts the topics of the population.

--------

<p><small>Project structure based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
