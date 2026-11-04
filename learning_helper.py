import torch
import time
import json
import os
from collections import Counter
from model import TextImageModel, TextOnlyModel, ImageOnlyModel


def majority_baseline(train_instances, test_instances):
    """
    get the results of majority baseline
    :param train_instances: training instances
    :param test_instances: test instances
    :return: a list of predicted labels
    """
    train_labels = [x['adjudicated_label'] for x in train_instances]
    majority_label = Counter(train_labels).most_common()[0][0]
    pred_labels = [majority_label] * len(test_instances)
    return pred_labels


def get_model(mode, annot_type, vision_api, para):
    """
    given the mode and the parameter dictionary, return the model
    :param mode: text_only OR image_only OR text_image
    :param annot_type: gold OR gold+silver
    :param para: parameter dictionary
    :return: model
    """
    if mode == 'text_only':
        bert_feat_dim = para['bert_feat_dim']
        hidden_dim = para['hidden_dim']
        dropout_rate = para['dropout_rate']
        output_dim = para['output_dim']
        flag_dim = para['flag_dim']
        if annot_type == 'gold+silver':
            model = TextOnlyModel(
                bert_feat_dim=bert_feat_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                output_dim=output_dim,
                flag_dim=flag_dim
            )
        else:
            model = TextOnlyModel(
                bert_feat_dim=bert_feat_dim,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                output_dim=output_dim,
                flag_dim=0
            )

    if mode == 'image_only':
        vgg_feat_dim = para['vgg_feat_dim']
        tag_feat_dim = para['tag_feat_dim']
        hidden_dim = para['hidden_dim']
        dropout_rate = para['dropout_rate']
        output_dim = para['output_dim']
        flag_dim = para['flag_dim']
        if annot_type == 'gold+silver':
            if vision_api:
                model = ImageOnlyModel(
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=tag_feat_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=flag_dim
                )
            else:
                model = ImageOnlyModel(
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=0,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=flag_dim
                )
        else:
            if vision_api:
                model = ImageOnlyModel(
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=tag_feat_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=0
                )
            else:
                model = ImageOnlyModel(
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=0,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=0
                )

    if mode == 'text_image':
        bert_feat_dim = para['bert_feat_dim']
        vgg_feat_dim = para['vgg_feat_dim']
        tag_feat_dim = para['tag_feat_dim']
        hidden_dim = para['hidden_dim']
        dropout_rate = para['dropout_rate']
        output_dim = para['output_dim']
        flag_dim = para['flag_dim']
        if annot_type == 'gold+silver':
            if vision_api:
                model = TextImageModel(
                    bert_feat_dim=bert_feat_dim,
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=tag_feat_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=flag_dim
                )
            else:
                model = TextImageModel(
                    bert_feat_dim=bert_feat_dim,
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=0,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=flag_dim
                )
        else:
            if vision_api:
                model = TextImageModel(
                    bert_feat_dim=bert_feat_dim,
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=tag_feat_dim,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=0
                )
            else:
                model = TextImageModel(
                    bert_feat_dim=bert_feat_dim,
                    vgg_feat_dim=vgg_feat_dim,
                    tag_feat_dim=0,
                    hidden_dim=hidden_dim,
                    dropout_rate=dropout_rate,
                    output_dim=output_dim,
                    flag_dim=0
                )

    return model


def train(classifier, iterator, optimizer, criterion, device, label_to_idx):
    """
    train the model
    :param classifier: the model to be trained
    :param iterator: training data loader
    :param optimizer: optimizer
    :param criterion: loss function
    :param device: cpu or cuda
    :param label_to_idx: convert the label into index to use the CrossEntropy
    :return: the loss and accuracy of each epoch
    """

    epoch_loss = 0
    epoch_correct = 0
    num_instances = 0

    for batch_x, batch_y in iterator:

        # get the text representation and image representation
        feats = batch_x.squeeze(1).to(device)

        label_ids = torch.tensor([label_to_idx[label] for label in batch_y]).to(device)

        optimizer.zero_grad()

        # use the model to predict
        preds = classifier(feats)
        preds = preds.squeeze(1).to(device)

        loss = criterion(preds, label_ids)
        epoch_loss += loss.item()

        # back-propogate
        loss.backward()
        optimizer.step()

        # calculate the accuracy
        pred_ids = preds.argmax(dim=1)
        epoch_correct += pred_ids.eq(label_ids).sum().item()
        num_instances += len(batch_y)

    return epoch_loss / num_instances, epoch_correct / num_instances


def evaluate(classifier, iterator, criterion, device, label_to_idx):
    """
    evaluate the trained model
    :param classifier: the trained model
    :param iterator: development data loader
    :param criterion: loss function
    :param device: cuda or cpu
    :param label_to_idx: convert the label into index to use the loss function
    :return: the loss and accuracy on development data
    """
    epoch_loss = 0
    epoch_correct = 0
    num_instances = 0

    classifier.eval()

    with torch.no_grad():

        for batch_x, batch_y in iterator:

            # get the text representation and image representation
            feats = batch_x.squeeze(1).to(device)

            label_ids = torch.tensor([label_to_idx[label] for label in batch_y]).to(device)

            # use the model to predict
            preds = classifier(feats)
            preds = preds.squeeze(1).to(device)

            loss = criterion(preds, label_ids)
            epoch_loss += loss.item()

            # calculate the accuracy
            pred_ids = preds.argmax(dim=1)
            epoch_correct += pred_ids.eq(label_ids).sum().item()
            num_instances += len(batch_y)

    return epoch_loss / num_instances, epoch_correct / num_instances


def epoch_time(start_time, end_time):
    """
    Compute the time spent for each epoch
    :param start_time: the time before each epoch
    :param end_time: the time after each epoch
    :return: the time spent for each epoch
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def loop_epochs(mode, model, num_epochs, patience, annot_type, train_loader, dev_loader, optimizer, criterion, device,
                label_to_idx, vision_api):
    """
    loop every epoch to train the model
    :param mode: text_only OR image_only OR text_image
    :param model: the model needs to be trained
    :param num_epochs: the maximum number of epochs
    :param patience: the parameter to stop the training earlier
    :param train_loader: the data loader of training set
    :param dev_loader: the data loader of development set
    :param optimizer: optimizer
    :param criterion: the loss function
    :param device: cpu OR cuda
    :param label_to_idx: convert label into the index to use the loss function
    :return: the trained model
    """
    best_valid_loss = float('inf')
    check_stopping = 0
    if vision_api:
        model_name = f'retrained_{mode}_{annot_type}_visionapi_classifier.pkl'
    else:
        model_name = f'retrained_{mode}_{annot_type}_classifier.pkl'
    for i in range(num_epochs):

        start_time = time.time()
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, device, label_to_idx)
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, device, label_to_idx)
        end_time = time.time()

        elapsed_mins, elapsed_secs = epoch_time(start_time, end_time)

        print("-" * 60)
        print(f"Epoch: {i+1} || Epoch Time: {elapsed_mins}m {elapsed_secs}s")
        print(f"Epoch: {i+1} || Train loss: {train_loss:.02f}, Train Acc: {train_acc:.02f}")
        print(f"Epoch: {i+1} || Dev loss: {dev_loss:.02f}, Dev Acc: {dev_acc:.02f}")

        # check if we need to save the model
        if dev_loss < best_valid_loss:
            check_stopping = 0
            best_valid_loss = dev_loss
            torch.save(model, model_name)
        else:
            check_stopping += 1
            print(f"The loss on development set does not decrease")
            if check_stopping == patience:
                print("The loss on development set does not decrease, stop training!")
                break

    return model


def test_model(model, test_loader, idx_to_label, device):
    """
    get the prediction using the trained model and test data
    :param model: trained model
    :param test_loader: the data loader of test data
    :param idx_to_label: convert the index to the label
    :return: the predicted labels
    """
    pred_labels = []
    for test_x, test_y in test_loader:
        with torch.no_grad():

            # get the text representation
            feats = test_x.squeeze(1).to(device)

            preds = model(feats)
            pred_ids = preds.argmax(dim=1)
            for pred_id in pred_ids:
                pred_label = idx_to_label[pred_id.item()]
                pred_labels.append(pred_label)

    return pred_labels


def save_prediction(mode, annot_type, vision_api, pred_labels, test_instances, prediction_dir):
    """
    save the predicted labels to do the error analysis
    :param mode: text_only OR image_only OR text_image
    :param pred_labels: predicted labels returned by train model
    :param test_instances: test instances
    :return:
    """
    # save the predicted label
    if vision_api:
        prediction_filename = f'prediction_{mode}_{annot_type}_visionapi.json'
    else:
        prediction_filename = f'prediction_{mode}_{annot_type}.json'
    prediction_filepath = os.path.join(prediction_dir, prediction_filename)
    with open(prediction_filepath, 'w+') as pred_file:
        for index, instance in enumerate(test_instances):
            # remove the bert_output and resnet_output
            if 'bert_output' in instance.keys():
                del instance['bert_output']
            if 'vgg_output' in instance.keys():
                del instance['vgg_output']
            if 'tag_bert_output' in instance.keys():
                del instance['tag_bert_output']
            # add the predicted label
            pred_label = pred_labels[index]
            instance['pred_label'] = pred_label
            # save the instance
            pred_file.write(json.dumps(instance))
            pred_file.write('\n')

