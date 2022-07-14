## denoiser_unet_test/cgan_trainer.py

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable

import os, json, copy, tqdm, hydra, mlflow, shutil

from denoier-test.code.scheduler import build_scheduler
from gli_aims.tools.mlops.aims_mlops import ManagedMLFlow

import numpy as np

"""
INPUT of GENERATOR : 8F Raw Image
OUTPUT of GENERATOR : 32F Dsn Image
INPUT of DISCRIMINATOR : 32F Syn Image or 32F Dsn Image
OUTPUT of DISCRIMINATOR : Classification wether the Image is Real 32F or Not
"""


def weights_init_normal(m):
  classname = m.__class__.__name__
  if classname.find("Conv") != -1:
    torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
  elif classname.find("BatchNorm2d") != -1:
    torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
    torch.nn.init.constant_(m.bias.data, 0.0)

class CGANTrainer(AIMSTrainer):
  def __init__(self, recipe_name, train_configuration, 
               mlops_configuration, model_configuration):
    super(CGANTrainer, self).__init__()
    self.train_configuration = train_configuration
    self.mlops_configuration = mlops_configuration
    self.model_configuration = model_configuration
    self.recipe_name = recipe_name

    self.current_loss_dict = None
    self.current_metric_dict = None
  
  def configuration(self):
    self.model_name = self.model_configuration['backbone']['name']
    assert self.model_name != 'CGAN', "This Model Should Not Be Trained on the CGAN Trainer"
    self.patch = (1, 512 //2 **4, 512 // 2 **4)
    self.G_model = Generator()
    self.D_model = Double_Discriminator() ## output : pred_1, pred_2, pred_3 (pred_1 + pred_2)

    self.G_optimizer = torch.optim.Adam(self.G_model.parameters())
    self.D_optimizer = torch.optim.Adam(self.D_model.parameters())

    self.G_scheduler = build_scheduler(
        self.train_configuration['scheduler'], self.G_optimizer
    )
    self.D_scheduler = build_scheduler(
        self.train_configuration['scheduler'], self.D_optimizer
    )

    strategy = self.train_configuration['strategy']
    self.epochs_random = strategy[0]['epoch'] ## 8F 4개중에서 임의로 하나를 선택하는 방법
    self.epochs_zero = strategy[1]['epoch'] ## 8F 4개중에서 어차피 실제 inference 단계에서는 첫번째를 사용하기 때문에 최종적으로는 zero
    self.epchs = self.epochs_random + self.epochs_zero

    self.G_loss = nn.MSELoss() ## MSE Loss (Pixel Level for 32F Image Generator)
    self.D_loss = nn.L1Loss() ## L1 Loss for Discriminator

    ## TODO : Make the Lamda Constants for Loss Instances more Flexible (Register Loss Function)
    self.loss_instance = [self.G_loss, self.D_loss]
    self.lamda = [1.0, 1.0]
 
    self.mlops = ManagedMLFlow(
        experiment_name = self.mlops_configuration['experiment_name'],
        run_name = self.mlops_configuration['run_name'],
        user_name = self.mlops_configuration['user_name'],
        tracking_uri = self.mlops_configuration['tracking_uri']
    )

    original_path = hydra.utils.get_original_cwd()
    ## saving the codes used for training
    self.mlops.save_artifacts(os.path.join(original_path, "denoiser_unet_test"))
    self.mlops.save_artifact(os.path.join(original_path, "aims_denoiser_trainer_entry.py"))
    self.mlops.log_param_dictionary(self.train_configuration)

    self.best_metric = None
    self.copy_inline_32f_datasets()
  
  def compute_loss_value(self, x, y, generator = True):
    if generator:
      instance = self.loss_instance[0];lamda = self.lamda[0]
    else:instance = self.loss_instance[1];lamda = self.lamda[1]
    return instance(x,y) * lamda
  
  def training_step_start(self):
    pass

  def training_epoch_start(self):
    self.model.train()
    self.losses = []
  
  def training_epoch_end(self):
    G_loss = torch.tensor([v['G_loss'] for v in self.losses]).mean().item()
    D_loss = torch.tensor([v['D_loss'] for d in self.losses]).mean().item()

    current_loss = torch.tensor([v['current-loss'] for v in self.losses]).mean().item()
    reference_loss = torch.tensor([v['reference-loss'] for v in self.losses]).mean().item()
    self.current_loss_dict = {
        'loss.training-G_loss' : G_loss,
        'loss.training-D_loss' : D_loss,
        'rel-metric.relative-loss' : reference_loss - current_loss,
        'rel-metric.current-loss' : current_loss,
        'rel-metric.reference-loss' : reference_loss
    }
    self.mlops.log_metric_dictionary(self.current_loss_dict, self.current_epoch)
  
  def update_current_dict(self, target_dict, current_dict):
    """
    Function for Updating the Best Metrics from the Prior Results
    """
    if target_dict is None:
      target_dict = current_dict
      target_dict.update({
          "best." + key : value for key, value in current_dict.items()
      })
    else:
      target_dict.update(current_dict)
      for key, value in current_dict.items():
        if target_dict['best.' + key] <= current_dict[key]:
          target_dict['best.' + key] = current_dict[key]
    return target_dict
  
  def training_step(self, batch):
    target_index = 0
    if self.current_epoch < self.epochs_random:
      target_index = np.random.randint(4)
    raw_8f_images = batch[target_index].cuda() ## Index of the 8F image for Input 
    synthesize_32f_images = batch[4].cuda() ## 32F Synthesized Image for Ground Truth
    raw_8f_paths = batch[5]

    ## Adversarial Ground Truths 
    valid = Variable(Tensor(np.ones((raw_8f_images.size(0), *self.patch))), requires_grad = False)
    fake = Variable(Tensor(np.zeros((raw_8f_images.size(0), *self.patch))), requires_grad = False)
    ## Generator
    denoised_32f_images = self.G_model(raw_8f_images)
    self.default_size = raw_8f_images.shape[-1]
    G_loss = self.compute_loss_value(denoised_32f_images,
                                   synthesized_32f_images, generator = True)
    ## FOR MONITORING ##
    intensity_loss = (synthesize_32f_images - denoised_32f_images).abs().mean(dim = (1, 2, 3)).mean().detach()
    reference_loss = (synthesize_32f_images - raw_8f_images).abs().mean(dim = (1, 2, 3)).mean().detach()

    ## Discriminator
    pred_1, pred_2 = self.D_model(denoised_32f_images, synthesize_32f_images) ## Fake, Real
    GAN_loss = self.compute_loss_value(pred_2, valid, generator = False)
    G_loss = GAN_loss + G_loss * 100

    D1_loss = self.compute_loss_value(pred_1, fake, generator = False)
    D2_loss = self.compute_loss_value(pred_2, valid, generator = False)

    D_loss = 0.5 * (D1_loss + D2_loss)

    loss_dict = {
        "G_loss" : G_loss, ## Generator Loss
        "D_loss" : D_loss, ## Discriminator Loss
        "current-loss" : intensity_loss,
        "reference-loss" : reference_loss
    }

    return loss_dict
  
  def training_step_end(self, loss_dict, epoch_tqdm):
    ## Train Generator
    self.G_optimizer.zero_grad()
    loss_dict['G_loss'].backward()
    self.G_optimizer.step()
    ## Train Discriminator
    self.D_optimizer.zero_grad()
    loss_dict['D_loss'].backward()
    self.D_optimizer.step()

    self.losses.append(loss_dict)

    epoch_tqdm.set_description(
            "[Train][Epoch %03d][Iter %04d]"
            " Generator Loss: %.5f | Disrciminator Loss : %.5f | Ref-Loss: %.5f | Diff: %.5f" %
            (self.current_epoch, self.current_step, loss_dict["G_loss"],
             loss_dict["D_loss"], loss_dict["reference-loss"],
             loss_dict["reference-loss"] - loss_dict["current-loss"]))
    
  def save(self, id_last = False):
    def convert_and_save_torch_model(model, prefix):
      clone_model = copy.deepcopy(self.model)
      sample_input = torch.zeros((1, 1, self.default_size, self.default_size)).cuda()
      clone_model.float();sample_input.float()
      os.makedirs("models", exist_ok = True)
      output_name = "models/" + prefix + self.recipe_name
      output_name += "-GPU"
      output_name += "-FULL"
      output_name += ".pt"

      traced_model = torch.jit.trace(clone_model, sample_input)
      torch.jit.save(traced_model, output_name)
      self.mlops.save_artifact(output_name, "results")
      denoiser_configuration = {
        "model": {
            "path": os.path.basename(output_name)
          },
         "input-data": {
            "data-type": "full",
            "normalization": {
              "type": "default-norm",
              "mean": 0.5,
               "std": 0.25
            }
         },
          "output-data": {
            "normalization": {
            "type": "default-norm",
            "mean": 0.5,
            "std": 0.25
          }
        }
      with open("models/AIMSDenoiser.json", "w", encoding="utf8") as f:
        json.dump(denoiser_configuration, f, indent=4)

      self.mlops.save_artifact("models/AIMSDenoiser.json", "results")
    
    state_dict = {
        'epoch' : self.current_epoch,
        'generator_state_dict' : self.G_model.state_dict(),
        'discriminator_state_dict' : self.D_model.state_dict()
    }
    current_metric = self.current_metric_dict['metric.dsn-vs-syn.rsq']
    if self.best_metric is None:
      self.best_metric = current_metric

    if self.best_metric <= current_metric:
      self.best_metric = current_metric
      torch.save(state_dict, "best.pth")
      convert_and_save_torch_model(self.G_model, "generator-best-")
      convert_and_save_torch_model(self.D_model, "discriminator-best-")
      self.mlops.save_artifact("best.pth", "results")
      self.rendering_samples()
      self.mlops.save_models()

    if is_last:
      torch.save(state_dict, "last.pth")
      convert_and_save_torch_model(self.G_model, "generator-last-")
      convert_and_save_torch_model(self.D_model, "discriminator-last-")
      self.mlops.save_artifact("last.pth", "results")
      self.rendering_samples()
      self.mlops.save_models()
    
  def rendering_samples(self):
    if not os.path.exists('./samples'):
      os.mkdir('./samples')
    with torch.no_grad():
      dataset = self.eval_data_loader.dataset
      for batch in self.eval_data_loader:
        raw_8f_images = batch[0].cuda()
        syn_32f_images = batch[4].cuda()
        raw_8f_pths = batch[5]
        denoised_32f_images = self.G_model(raw_8f_images)
        dataset.save_tensor(
          "samples/sample-%04d-raw-8f.png" % (self.current_epoch),raw_8f_images[0])
        dataset.save_tensor(
          "samples/sample-%04d-dsn-32f.png" % (self.current_epoch), denoised_32_images[0])
        dataset.save_tensor(
          "samples/sample-%04d-syn-32f.png" % (self.current_epoch), syn_32f_images[0])
        # TODO: should be improved to reduce the redundant copy
        self.mlops.save_artifacts("samples", "samples")
        break
  
  def evaluation(self):
    self.G_model.eval()
    self.generate_dataset()
    stats = self.evaluator.run()  
    current_metric_dict = {
      "metric.raw-vs-syn.rsq": stats["RAW"]["evaluation"]["linear-regression-with-intercept"]["rsq"],
      "metric.raw-vs-syn.rmse": stats["RAW"]["evaluation"]["rmse"],
      "metric.raw-vs-syn.mean-of-delta": stats["RAW"]["evaluation"]["mean-of-delta"],
      "metric.dsn-vs-syn.rsq": stats["DSN"]["evaluation"]["linear-regression-with-intercept"]["rsq"],
      "metric.dsn-vs-syn.rmse": stats["DSN"]["evaluation"]["rmse"],
      "metric.dsn-vs-syn.mean-of-delta": stats["RAW"]["evaluation"]["mean-of-delta"],
     }

    self.current_metric_dict = self.update_current_dict(
      self.current_metric_dict, current_metric_dict)
      self.mlops.log_metric_dictionary(self.current_metric_dict, self.current_epoch)
    return current_metric_dict["metric.dsn-vs-syn.rsq"]


  def copy_inline_32f_datasets(self, root_path="./datasets/"):
    inline_paths = self.eval_data_loader.dataset.inline_paths
    for inline_path in inline_paths:
      for dest in ["outputs-raw/.", "outputs-dsn/.", "outputs-syn/."]:
        output_path = os.path.join(root_path, dest)
        os.makedirs(output_path, exist_ok=True)
        shutil.copy(inline_path, output_path)
  
  def generate_datasets(self, root_path="./datasets/"):

    def save_images(image_tensors, paths, root_path, output_folder):
      for k in range(image_tensors.shape[0]):
        folder_name = os.path.basename(os.path.dirname(paths[k]))
        output_path = os.path.join(
        root_path, "%s/%s" % (output_folder, folder_name))
        if not os.path.exists(output_path):
          os.makedirs(output_path, exist_ok=True)
        image_index = int(os.path.basename(paths[k])[5:9])
        self.eval_data_loader.dataset.save_tensor(
        os.path.join(output_path,"S24_M%04d-02MS.png" % (image_index)),image_tensors[k])

    with torch.no_grad():
      for batch in tqdm.tqdm(self.eval_data_loader):
        raw_8f_images = batch[0]
        synthesize_32f_images = batch[4].cuda()
        raw_8f_paths = batch[5]
        denoised_32_images = self.G_model(batch[0].cuda()) ## Denoised Image is the Output of the Generative Model

        save_images(raw_8f_images, raw_8f_paths, root_path,
                            "outputs-raw")
        save_images(denoised_32_images, raw_8f_paths, root_path,
                            "outputs-dsn")
        save_images(synthesize_32f_images, raw_8f_paths, root_path,
                            "outputs-syn")


  
  

    








    
