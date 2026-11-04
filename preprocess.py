import json
import os
import time
import torch
import numpy as np
import transformers as ppb
import torchvision.models as models
from PIL import Image
from torchvision import transforms
from learning_helper import epoch_time
from torch.utils.data import Dataset, DataLoader


def check_argument(data_type, annot_type, mode, model_type):
    """

    :param data_type:
    :param annot_type:
    :param mode:
    :param model_type:
    :return:
    """
    res = [True, '']
    if data_type not in ['full', 'noimage', 'notext']:
        res[0] = False
        res[1] = 'data_type should be full OR noimage OR notext'
    if annot_type not in ['gold', 'silver', 'gold+silver']:
        res[0] = False
        res[1] = 'annotation_type should be gold OR silver OR gold+silver'
    if mode not in ['text_only', 'image_only', 'text_image', 'majority']:
        res[0] = False
        res[1] = 'mode should be text_only OR image_only OR text_image'
    if mode not in ['majority']:
        if model_type not in ['retrain', 'pretrained']:
            res[0] = False
            res[1] = 'model_type should be retrain OR pretrained'
    return res


def load_data(saved_anchor_dir, data_type, annotation_type, add_annot_type):
    """
    load the data
    :param saved_anchor_dir:
    :param data_type: full or noimage or notext
    :param annotation_type: gold or silver
    :param add_annot_type: add info about the annotation type or not
    :return: a list of dictionaries containing the essential information of all instances
    """

    # get the paths of the directories of annotation, image, json, and tag
    annotation_dir = os.path.join(saved_anchor_dir, 'annotation')
    image_dir = os.path.join(saved_anchor_dir, 'image_file')
    json_dir = os.path.join(saved_anchor_dir, 'json_file')
    tag_dir = os.path.join(saved_anchor_dir, 'tag_file')

    # get the paths of annotation file
    annotation_filename = f"{data_type}_{annotation_type}.json"
    annotation_filepath = os.path.join(annotation_dir, annotation_filename)

    # load the annotation file and tweet json file
    instances = []
    with open(annotation_filepath, 'r') as annot_jsonfile:
        lines = annot_jsonfile.read().split("\n")[:-1]
        for line in lines:
            instance = json.loads(line)

            # add the tweet text into the dictionary
            json_filename = instance['json_filename']
            json_filepath = os.path.join(json_dir, json_filename)
            with open(json_filepath, 'r') as jsonfile:
                tweet = json.loads(jsonfile.read())
            tweet_text = tweet['full_text']
            instance['tweet_text'] = tweet_text

            # add the image filepath into the dictionary
            image_filename = instance['image_filename']
            image_filepath = os.path.join(image_dir, image_filename)
            instance['image_filepath'] = image_filepath

            # add tags from google vision api
            tag_filename = f"tags_{image_filename.split('.')[0]}.json"
            tag_filepath = os.path.join(tag_dir, tag_filename)
            with open(tag_filepath, 'r') as tagfile:
                tag_file = json.load(tagfile)
            # for now, just use what we got from vision api
            tag_list = [x['tag'] for x in tag_file]
            tag_sequence = ', '.join(tag_list)
            instance['tag_sequence'] = tag_sequence

            # add the annotation type so that we can add binary flag embedding
            if add_annot_type:
                instance['annot_type'] = annotation_type

            instances.append(instance)

    return instances


def split_instances(instances, data_type, annotation_type):
    """
    split the instances into training part, development part, and test part
    :param instances: the instances loaded (needs to be split)
    :param data_type: full or noimage or notext
    :param annotation_type: gold or silver
    :return: three lists of instances (training, development, and test)
    """
    # load the split file
    split_filename = 'train_dev_test_split'
    with open(split_filename, 'r') as splitFile:
        split_dict = json.loads(splitFile.read())

    # load the anchor id of train, dev, and test
    type_ = f"{data_type}_{annotation_type}"
    all_annots = split_dict[type_]
    train_ids = all_annots['train']
    dev_ids = all_annots['dev']
    test_ids = all_annots['test']

    # load the instances of train, dev, and test
    train_instances = []
    dev_instances = []
    test_instances = []
    for instance in instances:
        if instance['anchor_id'] in train_ids:
            train_instances.append(instance)
        if instance['anchor_id'] in dev_ids:
            dev_instances.append(instance)
        if instance['anchor_id'] in test_ids:
            test_instances.append(instance)

    return train_instances, dev_instances, test_instances


def add_bert_output(instances):
    """
    get the input to the BERT and add it into the instance dictionary
    :param instances: the instance dictionary before adding the BERT input
    :return: a new instance dictionary which contains the BERT input
    """
    # load the BERT model and the tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_class, bert_tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    bert_model = bert_model_class.from_pretrained(pretrained_weights).to(device)
    tokenizer = bert_tokenizer_class.from_pretrained(pretrained_weights)

    # tokenize the tweet text
    text_tokenized = [tokenizer.encode(x['tweet_text'], add_special_tokens=True) for x in instances]

    # pad the sentences and prepare the mask for using the ids
    max_len = max([len(x) for x in text_tokenized])
    all_padded = np.array([i + [0] * (max_len - len(i)) for i in text_tokenized])
    all_attention_mask = np.where(all_padded != 0, 1, 0)

    # convert the id and mask into tensor
    all_padded_ids = torch.tensor(all_padded)
    all_attention_mask = torch.tensor(all_attention_mask)

    # add the bert input into the passed argument -- instances
    for index, instance in enumerate(instances):
        with torch.no_grad():
            small_padded_ids = all_padded_ids[index].unsqueeze(0).to(device)
            small_attention_mask = all_attention_mask[index].unsqueeze(0).to(device)
            last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

        # after getting the representation, move the tensor to CPU to free GPU memory
        bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
        instance['bert_output'] = bert_feature

    return instances


def add_tag_feat(instances):
    """
    get the input to the BERT and add it into the instance dictionary
    :param instances:
    :return:
    """
    # load the BERT model and the tokenizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bert_model_class, bert_tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')
    bert_model = bert_model_class.from_pretrained(pretrained_weights).to(device)
    tokenizer = bert_tokenizer_class.from_pretrained(pretrained_weights)

    # tokenize the tweet text
    text_tokenized = [tokenizer.encode(x['tag_sequence'], add_special_tokens=True) for x in instances]

    # pad the sentences and prepare the mask for using the ids
    max_len = max([len(x) for x in text_tokenized])
    all_padded = np.array([i + [0] * (max_len - len(i)) for i in text_tokenized])
    all_attention_mask = np.where(all_padded != 0, 1, 0)

    # convert the id and mask into tensor
    all_padded_ids = torch.tensor(all_padded)
    all_attention_mask = torch.tensor(all_attention_mask)

    # add the bert input into the passed argument -- instances
    for index, instance in enumerate(instances):
        with torch.no_grad():
            small_padded_ids = all_padded_ids[index].unsqueeze(0).to(device)
            small_attention_mask = all_attention_mask[index].unsqueeze(0).to(device)
            last_hidden_states = bert_model(small_padded_ids, attention_mask=small_attention_mask)

        # after getting the representation, move the tensor to CPU to free GPU memory
        bert_feature = last_hidden_states[0][:, 0, :].to('cpu')
        instance['tag_bert_output'] = bert_feature

    return instances


def add_vgg_output(instances):
    """
    add the image representation (from VGG16) into the dictionary
    :param instances: a list of instances need to be updated
    :return: a new new list of instances that are already updated
    """
    # Use the GPU, if available, to get the image representation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # get the pretrained VGG16 (discard the last classification layer)
    vgg16_model = models.vgg16(pretrained=True)
    for p in vgg16_model.parameters():
        p.requires_grad = False
    vgg16_model = vgg16_model.to(device)
    vgg16_model.eval()

    for instance in instances:
        # get the path of image file
        filepath = instance['image_filepath']

        # preprocess the image, convert it into RGB format if it is not RGB
        input_image = Image.open(filepath).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(input_image).unsqueeze(0).to(device)

        # get the image representation
        with torch.no_grad():
            output = vgg16_model(input_tensor)
            # after getting the image representation, move the tensor to the CPU to free GPU memory
            output = output.to('cpu')

        # add image representation into the dictionary
        instance['vgg_output'] = output

    return instances


def preprocess_instances(instances, mode, vision_api):
    """
    preprocess the instances, add bert/resnet output to the instance dictionary
    :param instances: the instances need to be enriched with text/image representation
    :param mode: text_only OR image_only OR text_image
    :return: the new instance dictionary
    """

    # add bert_output if the model needs textual information
    if mode == 'text_only':
        print("Preprocessing the text via pretrained BERT-base model...")
        bert_start_time = time.time()
        instances = add_bert_output(instances)
        bert_end_time = time.time()
        print("Done.")
        elapsed_mins, elapsed_secs = epoch_time(bert_start_time, bert_end_time)
        print(f"Time spent for BERT preprocessing: {elapsed_mins}m {elapsed_secs}s")

    # add VGG16 output of image and BERT output of tag sequence
    if mode == 'image_only':
        print("Preprocessing the image via pretrained VGG16 model and BERT-base model...")
        bert_start_time = time.time()
        instances = add_vgg_output(instances)
        if vision_api:
            instances = add_tag_feat(instances)
        bert_end_time = time.time()
        print("Done.")
        elapsed_mins, elapsed_secs = epoch_time(bert_start_time, bert_end_time)
        print(f"Time spent for BERT&VGG16 preprocessing: {elapsed_mins}m {elapsed_secs}s")

    # add VGG16 output of image and BERT output of tag sequence and tweet text
    if mode == 'text_image':
        print("Preprocessing the text and image via pretrained BERT-base model and VGG16 model...")
        vgg_start_time = time.time()
        instances = add_bert_output(instances)
        instances = add_vgg_output(instances)
        if vision_api:
            instances = add_tag_feat(instances)
        vgg_end_time = time.time()
        print("Done.")
        elapsed_mins, elapsed_secs = epoch_time(vgg_start_time, vgg_end_time)
        print(f"Time spent for BERT&VGG16 preprocessing: {elapsed_mins}m {elapsed_secs}s")

    return instances


def get_final_feats(instance):

    # BERT
    if 'bert_output' in instance.keys() and \
            'vgg_output' not in instance.keys() and \
            'tag_bert_output' not in instance.keys():
        feat = instance['bert_output']

    # VGG
    if 'vgg_output' in instance.keys() and \
            'bert_output' not in instance.keys() and \
            'tag_bert_output' not in instance.keys():
        feat = instance['vgg_output']

    # Vision API
    if 'tag_bert_output' in instance.keys() and \
            'bert_output' not in instance.keys() and \
            'vgg_output' not in instance.keys():
        feat = instance['tag_bert_output']

    # BERT + VGG
    if 'bert_output' in instance.keys() and \
            'vgg_output' in instance.keys() and \
            'tag_bert_output' not in instance.keys():
        bert_output = instance['bert_output']
        vgg_output = instance['vgg_output']
        feat = torch.cat((bert_output, vgg_output), dim=1)

    # BERT + Vision API
    if 'bert_output' in instance.keys() and \
            'tag_bert_output' in instance.keys() and \
            'vgg_output' not in instance.keys():
        bert_output = instance['bert_output']
        tag_feat = instance['tag_bert_output']
        feat = torch.cat((bert_output, tag_feat), dim=1)

    # VGG + Vision API
    if 'vgg_output' in instance.keys() and \
            'tag_bert_output' in instance.keys() and \
            'bert_output' not in instance.keys():
        vgg_output = instance['vgg_output']
        tag_feat = instance['tag_bert_output']
        feat = torch.cat((vgg_output, tag_feat), dim=1)

    # BERT + VGG + Vision API
    if 'bert_output' in instance.keys() and \
            'vgg_output' in instance.keys() and \
            'tag_bert_output' in instance.keys():
        bert_output = instance['bert_output']
        vgg_output = instance['vgg_output']
        tag_feat = instance['tag_bert_output']
        feat = torch.cat((bert_output, vgg_output, tag_feat), dim=1)

    return feat


class TweetDataset(Dataset):
    # write a new class to load our dataset
    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, index):
        instance = self.instances[index]
        label = instance['adjudicated_label']
        feat = get_final_feats(instance)
        # if instance has features from resnet and bert, then concatenate these two kinds of features
        gold_embedding = torch.tensor([[1, 0]])
        silver_embedding = torch.tensor([[0, 1]])
        if 'annot_type' in instance.keys():
            if instance['annot_type'] == 'gold':
                feat = torch.cat((feat, gold_embedding), dim=1)
            if instance['annot_type'] == 'silver':
                feat = torch.cat((feat, silver_embedding), dim=1)
        return feat, label

    def __len__(self):
        return len(self.instances)


def get_data_loader(train_instances, dev_instances, test_instances, batch_size):
    """
    get the data loaders for train, development, and test
    :param train_instances: the training data
    :param dev_instances: the development data
    :param test_instances: the test data
    :param batch_size: batch_size
    :return: data loaders for train, development, and test
    """
    train_dataset = TweetDataset(train_instances)
    dev_dataset = TweetDataset(dev_instances)
    test_dataset = TweetDataset(test_instances)

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, dev_loader, test_loader
