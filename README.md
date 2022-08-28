Are People Located in the Places They Mention in Their Tweets? 
A Multimodal Approach

## Requirements
python 3.6 \
Python packages: list of packages are provided in env_setup/requirements.txt file. \
Nvidia RTX 2060 \
CUDA 10.2 \
Pytorch 1.6

```
# Assume Nvidia RTX 2060 is available in your machine and CUDA 10.2 is correctly installed, then use 
    
python3 -m venv loctmp
source loctmp/bin/activate
pip install -r env_setup/requirements.txt

# Create the folder to store the prediction
mkdir predictions
``` 

## Experiments
The neural network model can take three kinds of data source:
1. Tweet text
2. Tweet image
3. Tweet text and image


Besides, you can use the training instances of the gold data to train the model, or, you can use the training instances \
of gold data and the instances of the silver data to train the model.


  - To get the experimental results with only the tweet text (tweet text)
    - ```python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode text_only --no-vision_api -model_type retrain```

  - To get the experimental results with only the image (tweet image)
    - ```python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode image_only --no-vision_api -model_type retrain```

  - To get the experimental results with only the tweet text and the image (text & image)
    - ```python main.py -data_dir saved_anchors -data_type full -annotation_type gold -mode text_image --no-vision_api -model_type retrain```


- Common problems:

1. **Data cannot be loaded correctly.** This is mostly because you do not set up environment correctly.

2. **Experimental results are SLIGHTLY different.** There are two potential reasons.
- a. There are some uncontrollable randomness in pytorch implementation, such as random seed of the packages we have used.
- b. GPU, CUDA, pytorch, and operation system are not compatible. We have conducted experiments with pytorch 1.6, CUDA 10.2 and Nvidia RTX 2060. Since for now we only have access to Nvidia RTX 3080 and the CUDA 10.2 is not compatiable with Nvidia RTX 3080, we are not able to confirm if this is the reason.


If you have any questions about the paper and code&data, please contact me via email: zhaominxiao@my.unt.edu
  
## Citation

If you want to use the corpus, please cite the paper below. (To be polished)
```bibtex
@inproceedings{,
    title = "Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach",
    author = "Xiao, Zhaomin and
      Blanco, Eduardo",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "",
    publisher = "Association for Computational Linguistics",
    url = "",
    pages = "",
    abstract = "",
}
```

This repository will contain the corpus of the COLING 2022 paper 
"Are People Located in the Places They Mention in Their Tweets? A Multimodal Approach"  
Authors: Zhaomin Xiao and Eduardo Blanco