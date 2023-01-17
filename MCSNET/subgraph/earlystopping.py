from common import logger
import os
import torch

class EarlyStoppingModule(object):
  """
    Module to keep track of validation score across epochs
    Stop training if score not imroving exceeds patience
  """  
  def __init__(self, av, patience=100, delta=0.0001):
    self.av = av
    self.patience = patience 
    self.delta = delta
    self.best_scores = None
    self.num_bad_epochs = 0 
    self.should_stop_now = False

  def save_latest_model(self, model, epoch,optimizer): 
    save_dir = os.path.join(self.av.DIR_PATH, "latestModels")
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    name = self.av.DATASET_NAME
    if self.av.TASK !="":
      name = self.av.TASK + "_" + name
    #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
    save_path = os.path.join(save_dir, name)

    #logger.info("saving best validated model to %s",save_path)
    output = open(save_path, mode="wb")
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch':epoch,
            'patience': self.patience,
            'best_scores': self.best_scores,
            'num_bad_epochs': self.num_bad_epochs,
            'should_stop_now': self.should_stop_now,
            'optim_state_dict': optimizer.state_dict(),
            'av' : self.av,
            }, output)
    output.close()

  def load_latest_model(self):
    load_dir = os.path.join(self.av.DIR_PATH, "latestModels")
    #if not os.path.isdir(load_dir):
    #  raise Exception('{} does not exist'.format(load_dir))
    name = self.av.DATASET_NAME
    if self.av.TASK !="":
      name = self.av.TASK + "_" + name
    #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
    load_path = os.path.join(load_dir, name)

    if not os.path.exists(load_path):
        return None

    logger.info("loading latest trained model from %s",load_path)
    checkpoint = torch.load(load_path)
    self.patience = checkpoint['patience']
    self.best_scores = checkpoint['best_scores']
    self.num_bad_epochs = checkpoint['num_bad_epochs']
    self.should_stop_now = checkpoint['should_stop_now']
    return checkpoint



  def save_best_model(self, model, epoch): 
    save_dir = os.path.join(self.av.DIR_PATH, "bestValidationModels")
    if not os.path.isdir(save_dir):
      os.makedirs(save_dir)
    name = self.av.DATASET_NAME
    if self.av.TASK !="":
      name = self.av.TASK + "_" + name
    #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
    save_path = os.path.join(save_dir, name)

    logger.info("saving best validated model to %s",save_path)
    output = open(save_path, mode="wb")
    torch.save({
            'model_state_dict': model.state_dict(),
            'epoch':epoch,
            'av' : self.av,
            }, output)
    output.close()

  def load_best_model(self):
    load_dir = os.path.join(self.av.DIR_PATH, "bestValidationModels")
    if not os.path.isdir(load_dir):
      raise Exception('{} does not exist'.format(load_dir))
    name = self.av.DATASET_NAME
    if self.av.TASK !="":
      name = self.av.TASK + "_" + name
    #name = name + "_tfrac_" + str(self.av.TEST_FRAC) + "_vfrac_" + str(self.av.VAL_FRAC)
    load_path = os.path.join(load_dir, name)
    logger.info("loading best validated model from %s",load_path)
    checkpoint = torch.load(load_path)
    return checkpoint

  def diff(self, curr_scores):
    return sum ([cs-bs for cs,bs in zip(curr_scores, self.best_scores)])

  def check(self,curr_scores,model,epoch,optimizer) :
    if self.best_scores==None: 
      self.best_scores = curr_scores
      self.save_best_model(model,epoch)
    elif self.diff(curr_scores) >= self.delta:
      self.num_bad_epochs = 0
      self.best_scores = curr_scores
      self.save_best_model(model,epoch)
    else:  
      self.num_bad_epochs+=1
      if self.num_bad_epochs>self.patience: 
        self.should_stop_now = True
    self.save_latest_model(model,epoch,optimizer)
    return self.should_stop_now  
