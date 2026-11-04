import json
import os
import sys
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from learning_helper import majority_baseline, get_model, loop_epochs, test_model, save_prediction
from preprocess import load_data, preprocess_instances, split_instances, get_data_loader, check_argument
from sklearn.metrics import classification_report

# control the source of randomness
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

pretrained_model_dir = 'pretrained_models'
prediction_dir = 'predictions'


def baselines(mode, annot_type, train_instances, dev_instances, test_instances):
    """
    get the results of baseline models
    :param mode:
    :param train_instances:
    :param dev_instances:
    :param test_instances:
    :return:
    """
    if mode == 'majority':
        train_instances = train_instances + dev_instances
        gold_labels = [x['adjudicated_label'] for x in test_instances]
        pred_labels = majority_baseline(train_instances, test_instances)
        print('-' * 60)
        print(classification_report(gold_labels, pred_labels))

    # save the predicted labels
    save_prediction(mode, annot_type, pred_labels, test_instances, prediction_dir)


def nn_method(mode, model_type, annot_type, vision_api, train_instances, dev_instances, test_instances):

    # get the label mapping
    label_to_idx = {'Yes': 1, 'Other': 0}
    idx_to_label = {1: 'Yes', 0: 'Other'}

    # load the parameter setting file
    para_filename = 'para_setting.json'
    with open(para_filename, 'r') as para_file:
        para_dict = json.loads(para_file.read())
    para = para_dict[mode]

    # get the data loader for train, development, and test
    batch_size = para['batch_size']
    train_loader, dev_loader, test_loader = get_data_loader(train_instances, dev_instances, test_instances, batch_size)

    # get the model (pretrained OR raw model)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if model_type == 'retrain':
        model = get_model(mode, annot_type, vision_api, para).to(device)
        print(model)

        # set the optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-03)
        criterion = nn.CrossEntropyLoss().to(device)

        # train and save the trained model
        num_epochs = para['num_epochs']
        patience = para['patience']
        model = loop_epochs(mode=mode, model=model, annot_type=annot_type, num_epochs=num_epochs, patience=patience,
                            train_loader=train_loader, dev_loader=dev_loader, vision_api=vision_api,
                            device=device, optimizer=optimizer, criterion=criterion, label_to_idx=label_to_idx)

    if model_type == 'pretrained':
        if vision_api:
            model_filename = f"pretrained_{mode}_{annot_type}_visionapi_classifier.pkl"
        else:
            model_filename = f"pretrained_{mode}_{annot_type}_classifier.pkl"
        model_path = os.path.join(pretrained_model_dir, model_filename)
        model = torch.load(model_path)

    # make prediction and print the classification report
    model = model.to(device)
    model.eval()
    pred_labels = test_model(model, test_loader, idx_to_label, device)
    gold_labels = [x['adjudicated_label'] for x in test_instances]
    print('-' * 60)
    print(classification_report(gold_labels, pred_labels))

    # save the predicted label
    save_prediction(mode, annot_type, vision_api, pred_labels, test_instances, prediction_dir)


def main(data_dir, data_type, annot_type, mode, model_type, vision_api):
    # check the argument
    check_result = check_argument(data_type, annot_type, mode, model_type)
    if not check_result[0]:
        sys.exit(check_result[1])

    # load the instances and preprocess the instances
    saved_anchor_dir = data_dir
    # saved_anchor_dir = '/home/zhaomin/Documents/saved_anchors'
    print('-' * 60 + '\n' + "Loading and preprocessing the tweets...")
    if annot_type != 'gold+silver':
        instances = load_data(saved_anchor_dir, data_type, annot_type, add_annot_type=False)
        instances = preprocess_instances(instances, mode, vision_api)
    else:
        instances = load_data(saved_anchor_dir, data_type, annotation_type='gold', add_annot_type=True)
        additional_instances = load_data(saved_anchor_dir, data_type, annotation_type='silver', add_annot_type=True)
        instances = preprocess_instances(instances, mode, vision_api)
        additional_instances = preprocess_instances(additional_instances, mode, vision_api)
    print("Done." + '\n' + '-' * 60)

    # split the instances into train, development, and test
    if annot_type == 'gold+silver':
        train_instances, dev_instances, test_instances = split_instances(instances, data_type, annotation_type='gold')
        train_instances = train_instances + additional_instances
        random.shuffle(train_instances)
    else:
        train_instances, dev_instances, test_instances = split_instances(instances, data_type, annot_type)

    # switch to neural method part
    if mode in ['text_only', 'image_only', 'text_image']:
        nn_method(mode, model_type, annot_type, vision_api, train_instances, dev_instances, test_instances)

    # switch to majority baseline
    if mode in ['majority']:
        baselines(mode, annot_type, train_instances, dev_instances, test_instances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-data_dir', '--dir', help='The path of the data folder')
    parser.add_argument('-data_type', '--data', help='Select the data type, full OR noimage OR notext')
    parser.add_argument('-annotation_type', '--annotation', help='Select the annotation type, gold OR silver')
    parser.add_argument('-mode', '--mode', help='Select the mode (text_only OR image_only OR text_image)')
    parser.add_argument('-model_type', '--type', help='Use pretrained model or retrained model')
    parser.add_argument('--vision_api', dest='vision_api', action='store_true')
    parser.add_argument('--no-vision_api', dest='vision_api', action='store_false')
    args = parser.parse_args()

    main(data_dir=args.dir, data_type=args.data, annot_type=args.annotation,
         mode=args.mode, model_type=args.type, vision_api=args.vision_api)
