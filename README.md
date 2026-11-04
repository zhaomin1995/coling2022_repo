# Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach

Corpus and code for the COLING 2022 paper
[Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach](https://aclanthology.org/2022.coling-1.226/)
by Zhaomin Xiao and Eduardo Blanco.

The corpus contains 6,540 tweets (text + image) annotated with whether the author
was located in the city mentioned in the tweet. Annotations were crowdsourced on
Amazon Mechanical Turk in three independent conditions and adjudicated with
[MACE](https://github.com/dirkhovy/MACE):

- `full`: annotators saw the original tweet (text and image)
- `noimage`: annotators saw only the text
- `notext`: annotators saw only the image

## Repository map

- **Just want the final annotations?** → `saved_anchors/annotation/`
- **Raw MTurk exports and MACE input/output?** → `mturk_annotation/`
- **Run the experiments?** → the commands below, from the repository root

## Repository layout

| Path | Contents |
|---|---|
| `main.py`, `model.py`, `preprocess.py`, `learning_helper.py` | experiment code |
| `use_vision_api.py` | script used to obtain Google Cloud Vision API tags |
| `para_setting.json` | hyperparameter settings |
| `train_dev_test_split` | train/dev/test anchor ids for every data/annotation variant |
| `predictions/` | model predictions on the test split (the systems in Table 2 of the paper, plus the majority baseline) |
| `saved_anchors/annotation/` | final adjudicated gold and silver annotations (`{full,noimage,notext}_{gold,silver}.json`, one JSON object per line) |
| `saved_anchors/json_file/` | tweet text, one trimmed tweet JSON per instance (see *Privacy and anonymization*) |
| `saved_anchors/tag_file/` | Google Cloud Vision API tags, one JSON per image |
| `mturk_annotation/batches_part1/` | raw MTurk batch CSVs and MACE input/output per annotation round: `small_batches/*.csv` (published batches and their `*_finished.csv` raw MTurk exports), `big.csv` (MACE input), `big.competence` / `big.prediction` (MACE output), `instances_keep_*.json` / `instances_republish_*.json` (round filtering) |
| `mturk_annotation/annotation_helper.py` | helper code used during annotation |
| `env_setup/requirements.txt` | Python dependencies |

### Images

The tweet screenshots (10,098 jpg, ~1.5 GB) are too large for this repository.
Download them separately and place them under `saved_anchors/image_file/`:

```
wget http://www.cse.unt.edu/~blanco/saved_anchors.zip
unzip saved_anchors.zip
```

Text-only experiments (`-mode text_only`) run without the images.

## Privacy and anonymization

- All MTurk WorkerIds (the `WorkerId` column of the CSV files and the
  `Answer.*_<WorkerId>` keys of the JSON files) have been replaced with
  consistent pseudonyms `annotator_0001`, `annotator_0002`, ... The same
  pseudonym refers to the same crowdworker across all files. The mapping to
  real WorkerIds is not distributed.
- The tweet JSON files in `saved_anchors/json_file/` keep only the fields
  `created_at`, `id`, `id_str`, `full_text`, and `lang`. User objects,
  mentions, and all other metadata were removed.
- The corpus contains one tweet per user, and the models only use the content
  of the tweet (text and image), never user information (see Section 1.1 of
  the paper). For takedown requests, please contact the authors.

## Environment setup

The code should be run on Linux with an Nvidia GPU (experiments were conducted
with pytorch 1.6, CUDA 10.2, and an Nvidia RTX 2060).

```
python3 -m venv loctmp
source loctmp/bin/activate
pip install -r env_setup/requirements.txt
```

## Experiments

Majority baseline:

```
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode majority
```

The neural network model can take five kinds of data source:
1. Tweet text,
2. The image included in the tweet,
3. Tweet text and the image,
4. The image and the corresponding tags returned by Google Cloud Vision API,
5. The tweet text, image, and the tags.

Train on the gold training instances (`-annotation_type gold`):

```
# 1. text only
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode text_only --no-vision_api -model_type retrain

# 2. image only
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode image_only --no-vision_api -model_type retrain

# 3. text and image
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode text_image --no-vision_api -model_type retrain

# 4. image and Vision API tags
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode image_only --vision_api -model_type retrain

# 5. text, image, and Vision API tags
python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode text_image --vision_api -model_type retrain
```

As a pilot study, we also experimented with training on the gold training
instances plus all silver instances (replace `-annotation_type gold` with
`-annotation_type gold+silver`). We did not keep this line of experiments in
the final paper.

Common problems:

1. **Data cannot be loaded correctly.** This is mostly because the environment
   is not set up correctly, or the images have not been downloaded (image modes).
2. **Experimental results are SLIGHTLY different.** There is some uncontrollable
   randomness in the pytorch implementation (random seeds of the packages used),
   and results can vary across GPU / CUDA / pytorch / OS combinations.

If you have any questions about the paper, code, or data, please contact me via
email: zhaominxiao@my.unt.edu

## Citation

If you use the corpus, please cite:

```bibtex
@inproceedings{xiao-blanco-2022-people,
    title = "Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach",
    author = "Xiao, Zhaomin  and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.226",
    pages = "2561--2571",
}
```
