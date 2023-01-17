import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
from scipy.stats import kendalltau
import itertools
from collections import defaultdict
import numpy as np

def evaluate(av,model,sampler):
  model.eval()

  pred = []
  targets = []

  n_batches = sampler.create_batches(shuffle=False)
  for i in range(n_batches):
    batch_data,batch_data_sizes,target,batch_adj = sampler.fetch_batched_data_by_id(i)
    pred.append( model(batch_data,batch_data_sizes,batch_adj).data)
    targets.append(target)
  all_pred = torch.cat(pred,dim=0)
  all_target = torch.cat(targets,dim=0)
  ndcg = ndcg_score([all_target.cpu().tolist()],[all_pred.cpu().tolist()])
  mse = torch.nn.functional.mse_loss(all_target, all_pred,reduction="mean").item()
  mae = torch.nn.functional.l1_loss(all_target, all_pred,reduction="mean").item()
  rankcorr = kendalltau(all_pred.cpu().tolist(),all_target.cpu().tolist())[0]

  #TODO: query wise stuff
  return ndcg,mse,rankcorr,mae


def pairwse_reward_similarity(scorePos, scoreNeg):
    """
      score* are similarity measures
      We reward +1 for every positive score > some negative score
    """
    scorePosExp = scorePos.unsqueeze(1)
    scoreNegExp = scoreNeg.unsqueeze(1)
    n_1 = scorePosExp.shape[0]
    n_2 = scoreNegExp.shape[0]
    dim = scorePosExp.shape[1]

    expanded_pos = scorePosExp.unsqueeze(1).expand(n_1, n_2, dim)
    expanded_neg = scoreNegExp.unsqueeze(0).expand(n_1, n_2, dim)
    ell = torch.sign(expanded_pos - expanded_neg)
    hinge = torch.nn.ReLU()
    loss = hinge(ell)
    sum_loss =  torch.sum(loss,dim= [0, 1])
    return sum_loss,(n_1*n_2)

