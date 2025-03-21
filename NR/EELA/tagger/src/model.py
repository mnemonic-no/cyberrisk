# -*- coding: utf-8 -*-

from src.metric import Metric
from src.utils import CEOR
import numpy as np
import torch, copy
import torch.nn as nn
from torch import log, exp
import torch.nn.functional as F
from scipy.stats import entropy as calc_entropy
from scipy.stats import kendalltau
from math import log, sqrt
from sklearn.metrics import f1_score, mean_squared_error, confusion_matrix



def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss


def ordinal_loss(output, target, alpha=2):
    probs = torch.nn.functional.softmax(output, dim=-1)
    p = probs.max(dim=-1).values
    loss = torch.neg(torch.sum(torch.log(torch.ones(target.shape)- p) * torch.pow(torch.abs(output.argmax(dim=-1)-target), alpha)))
    return loss

def ordinal_loss2(output, target, alpha=2):
    #probs = torch.nn.functional.softmax(output, dim=-1)
    #p = probs.max(dim=-1).values
    loss = torch.sum(torch.pow(torch.abs(output.argmax(dim=-1).float().requires_grad_()-target), alpha))
    return loss

class Model(object):

    def __init__(self, vocab, tagger):
        super(Model, self).__init__()

        self.vocab = vocab
        self.tagger = tagger
        label_tags = copy.deepcopy(self.vocab.tags)
        self.eval_labels = self.vocab.tag2id(label_tags)
        self.criterion = nn.CrossEntropyLoss()
        #self.criterion = ordinal_loss
        
        
    def calc_label_accuracy(self, g_enc_all, p_enc_all, evaluate=False):
        correct = 0.
        tokens = 0.
        gold, pred = [], []
        off_by_1, off_by_2 = 0, 0
        sum_distance = 0
        for g_enc, p_enc in zip(g_enc_all, p_enc_all):
            for g, p in zip(g_enc, p_enc):
                d = g.item() - p.item()
                if g.item() == p.item():
                    correct += 1.
                if np.absolute(d) <= 1:
                    off_by_1 += 1
                if  np.absolute(d) <= 2:
                    off_by_2 += 1
                tokens += 1
                gold.append(g.item())
                pred.append(p.item())
                sum_distance += d
        f1 =  f1_score(gold, pred, average='weighted')#, labels=self.eval_labels, zero_division=np.nan)
        mse = mean_squared_error(gold, pred)
        print("f1:", format(f1*100,".2f"))
        print("ob1", format(off_by_1/tokens*100, ".2f"))
        print("ob2", format(off_by_2/tokens*100, ".2f"))
        print("Mdist(neg=overestimate p_exp)", format(sum_distance/tokens, ".3f"))
        f1_scores = f1_score(gold, pred, average=None, labels=self.eval_labels, zero_division=np.nan)
        labels = self.vocab.id2tag([el.item() for el in self.eval_labels])
        f1_scores_with_labels = {label:format(score*100, ".2f") for label,score in zip(labels, f1_scores)}
        if evaluate:
            print(f1_scores_with_labels)
            print(set(gold))
            print(confusion_matrix(gold, pred, normalize="true"))
            print("ktau", kendalltau(gold, pred))
        return f1 #mse
    
    def train(self, loader):
        self.tagger.train()
        
        for words, chars, flags, tags in loader:
            self.optimizer.zero_grad()
            mask = flags.ne(self.vocab.pad_index) #cve_flag_dict["cve"])
            #new_tags = tags.float()
            #mask = words.ne(self.vocab.pad_index)
            #mask = flags.eq(self.vocab.cve_flag_dict["cve"])
            # ignore the first token of each sentence
            mask[:, 0] = 0
            #print(self.vocab.tag_dict)
            #print(flags)
            gold_tags = tags[mask]
            #print(gold_tags)
            #exit()
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            loss = self.get_loss(s_tag, gold_tags)
            loss.backward()
            nn.utils.clip_grad_norm_(self.tagger.parameters(), 5.0)
            self.optimizer.step()
            self.scheduler.step()

    def get_loss(self, s_tags, gold_tags):
        loss = self.criterion(s_tags,
                              gold_tags)
        return loss

    
    @torch.no_grad()
    def evaluate(self, loader, punct=True, evaluate=False):
        self.tagger.eval()

        loss, acc = 0, 0.
        all_pred = []
        all_gold = [] 
        for words, chars, flags, tags in loader:
            mask = flags.ne(self.vocab.pad_index)
            #mask = flags.eq(self.vocab.cve_flag_dict["cve"])
            #mask = words.ne(self.vocab.pad_index)
            #mask = flags.ne(self.vocab.cve_flag_dict["not_cve"])
            # ignore the first token of each sentence
            #mask[:, 0] = 0
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            gold_tags = tags[mask]
            pred_tag = self.decode(s_tag)
            all_gold.append(gold_tags)
            all_pred.append(pred_tag)
            loss += self.get_loss(s_tag,gold_tags)
        acc = self.calc_label_accuracy(all_gold, all_pred, evaluate=evaluate)
        loss /= len(loader)
        return loss, acc

    @torch.no_grad()
    def predict(self, loader):
        self.tagger.eval()

        all_tags, all_rels = [], []
        for words, chars, flags in loader:
            mask = flags.ne(self.vocab.pad_index)
            #mask = flags.eq(self.vocab.cve_flag_dict["cve"])
            #mask = words.ne(self.vocab.pad_index)
            # ignore the first token of each sentence
            mask[:, 0] = 0
            #mask = flags.ne(self.vocab.cve_flag_dict["not_cve"])
            lens = mask.sum(dim=1).tolist()
            s_tag = self.tagger(words, chars)
            s_tag = s_tag[mask]
            pred_tag = self.decode(s_tag)
            all_tags.extend(torch.split(pred_tag, lens))
        all_tags = [self.vocab.id2tag(seq) for seq in all_tags]
        return all_tags

    def decode(self, s_tag):
        pred_tag =  s_tag.argmax(dim=-1)
        return pred_tag
