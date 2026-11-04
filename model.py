import torch.nn as nn
import torch.nn.functional as F
import torch


class TextImageModel(nn.Module):

    def __init__(self, bert_feat_dim, vgg_feat_dim, tag_feat_dim, output_dim, hidden_dim, dropout_rate, flag_dim):
        super().__init__()
        self.bert_feat_dim = bert_feat_dim
        self.vgg_feat_dim = vgg_feat_dim
        self.tag_feat_dim = tag_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.flag_dim = flag_dim
        self.fc_bert = nn.Linear(bert_feat_dim, hidden_dim)
        self.fc_vgg = nn.Linear(vgg_feat_dim, hidden_dim)
        self.fc_both = nn.Linear(bert_feat_dim+vgg_feat_dim, hidden_dim)
        self.fc_last1 = nn.Linear(hidden_dim*3+flag_dim, hidden_dim)
        self.fc_last2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, feat_combined):
        bert_feat = feat_combined[:, :self.bert_feat_dim]
        vgg_feat = feat_combined[:, self.bert_feat_dim:self.bert_feat_dim+self.vgg_feat_dim]
        both_feat = torch.cat((bert_feat, vgg_feat), dim=1)
        bert_out = self.fc_bert(bert_feat)
        vgg_out = self.fc_vgg(vgg_feat)
        both_out = self.fc_both(both_feat)
        if feat_combined.shape[1] > self.bert_feat_dim+self.vgg_feat_dim:
            flag_feat = feat_combined[:, self.bert_feat_dim+self.vgg_feat_dim:]
            combined_out = torch.cat((bert_out, vgg_out, both_out, flag_feat), dim=1)
        else:
            combined_out = torch.cat((bert_out, vgg_out, both_out), dim=1)
        out = self.fc_last1(combined_out)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.fc_last2(out)
        return out


class TextOnlyModel(nn.Module):

    def __init__(self, bert_feat_dim, output_dim, hidden_dim, dropout_rate, flag_dim):
        super().__init__()
        self.bert_feat_dim = bert_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.flag_dim = flag_dim
        self.combined_feat_dim = bert_feat_dim + flag_dim
        self.fc1 = nn.Linear(self.combined_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, bert_feat):
        out = self.fc1(bert_feat)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out


class ImageOnlyModel(nn.Module):

    def __init__(self, vgg_feat_dim, tag_feat_dim, output_dim, hidden_dim, dropout_rate, flag_dim):
        super().__init__()
        self.vgg_feat_dim = vgg_feat_dim
        self.tag_feat_dim = tag_feat_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.flag_dim = flag_dim
        self.combined_feat_dim = vgg_feat_dim + tag_feat_dim + flag_dim
        self.fc1 = nn.Linear(self.combined_feat_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, vgg_feat):
        out = self.fc1(vgg_feat)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
