## denoiser_unet_test/jihye_denoiser_trainer.py

import os, json, copy, tqdm, hydra, torch, mlflow, shutil, torch
from base_trainer import BaseTrainer

def update_current_dict(targ, curr):
  if targ is None:
    targ = curr
    targ.update(
        {"best." + key:value for key, value in curr.items()}
    )
  else:
    targ.update(curr)
    for key, value in curr.items():
      if targ['best.' + key] <= curr[key]:
        targ['best.' + key] = curr[key]

  return targ

class DenoiserEvaluator:
  def __init__(self, cfg, evaluator, dataloader, current_metric_dict):
    self.tool = evaluator
    self.cfg = cfg
    self.eval_data_loader = dataloader
    self.current_metric_dict = current_metric_dict

  def evaluate(self, model, current_epoch):
    model.eval()
    self.generate_dataset(model)
    stats = self.tool.run()
    current_metric_dict = {
            "metric.raw-vs-syn.rsq":
                stats["RAW"]["evaluation"]["linear-regression-with-intercept"]
                ["rsq"],
            "metric.raw-vs-syn.rmse":
                stats["RAW"]["evaluation"]["rmse"],
            "metric.raw-vs-syn.mean-of-delta":
                stats["RAW"]["evaluation"]["mean-of-delta"],
            "metric.dsn-vs-syn.rsq":
                stats["DSN"]["evaluation"]["linear-regression-with-intercept"]
                ["rsq"],
            "metric.dsn-vs-syn.rmse":
                stats["DSN"]["evaluation"]["rmse"],
            "metric.dsn-vs-syn.mean-of-delta":
                stats["RAW"]["evaluation"]["mean-of-delta"],
        }
    self.current_metric_dict = update_current_dict(
        self.current_metric_dict, current_metric_dict
    )
    self.cfg.mlops.log_metric_dictionary(self.current_metric_dict, current_epoch)
    
    ## Most Important Metric for Evaluating the Quality of the Denoiser
    return current_metric_dict['metric.dsn-vs-syn.rsq']

  def generate_dataset(self, model, root_path = './datasets/'):
    with torch.no_grad():
      for batch in tqdm.tqdm(self.eval_data_loader):
        raw_8f_images = batch[0]
        synthesize_32f_images = batch[4].cuda()
        raw_8f_pths = batch[5]
        denoised_32f_images = model(batch[0].cuda())
        self.save_images(raw_8f_images, raw_8f_pths, root_path, 'outputs-raw')
        self.save_images(denoised_32f_images, raw_8f_pths, root_path, 'outputs-dsn')
        self.save_images(synthesize_32f_images, raw_8f_pths, root_path, 'outputs-syn')
  
  def save_images(self, img, pth, root, output_folder):
    B = img.shape[0]
    for k in range(B):
      folder_name = os.path.basename(os.path.dirname(pth[k]))
      output_pth = os.path.join(root, "%s/%s" % (output_folder, folder_name))
      if not os.path.exists(output_pth):
        os.makedirs(output_pth, exist_ok = True)
      img_idx = int(os.path.basename(pth[k])[5:9])
      self.eval_data_loader.dataset.save_tensor(
        os.path.join(output_pth, "S24_M%04d-02MS.png" % (img_idx)),
                    img[k])
  

class DenoiserTrainer:
  def __init__(self, cfg_instance):
    self.cfg = cfg_instance
    self.model = self.cfg.model
    self.current_metric_dict = None
    self.current_loss_dict = None

  def run(self, train_data_loader, eval_data_loader, eval_configuration, evaluator = None):
    self.train_data_loader = train_data_loader
    self.eval_data_loader = eval_data_loader
    self.evaluator = DenoiserEvaluator(self.cfg, evaluator, self.eval_data_loader, self.current_metric_dict)
    self.eval_epochs = eval_configuration['epoch']

    inline_paths = self.eval_data_loader.dataset.inline_paths
    root_path = './datasets/'
    for inline_path in inline_paths:
      for dest in ["outputs-raw/.", "outputs-dsn/.", "outputs-syn/."]:
        output_path = os.path.join(root_path, dest)
        os.makedirs(output_path, exist_ok=True)
        shutil.copy(inline_path, output_path)

    for epoch in range(self.cfg.epochs):
      self.current_epoch = epoch
      epoch_tqdm = tqdm(enumerate(train_data_loader), ncols = 80)
      self.training_epoch_start()

      for step, batch in epoch_tqdm:
        self.current_step = step
        self.training_step_start()
        loss = self.training_step(batch)
        self.training_step_end(loss, epoch_tqdm)

      self.training_epoch_end()

      if (epoch + 1) % self.eval_epochs == 0:
        self.evaluator.evaluate(self.model, self.current_epoch)
        self.save(False)

    ret = self.evaluator.evaluate(self.model, self.current_epoch)
    self.save(True)
    return ret
  
  def training_epoch_start(self):
    self.model.train() ## make gradient update available
    self.losses = [] ## list to save the training loss dictionary

  def training_step_start(self):
    pass
  
  def training_step_end(self, loss_dict, epoch_tqdm):
    self.cfg.optimizer.zero_grad()
    loss_dict['loss'].backward()
    self.cfg.optimizer.step()
    self.losses.append(loss_dict)
    
    epoch_tqdm.set_description(
            "[Train][Epoch %03d][Iter %04d]"
            " Loss: %.5f | Base-Loss: %.5f | Ref-Loss: %.5f | Diff: %.5f" %
            (self.current_epoch, self.current_step, loss_dict["loss"],
             loss_dict["current-loss"], loss_dict["reference-loss"],
             loss_dict["reference-loss"] - loss_dict["current-loss"]))
  
  def training_epoch_end(self):
    loss = torch.tensor([v["loss"] for v in self.losses]).mean().item()
    current_loss = torch.tensor([v["current-loss"] for v in self.losses]).mean().item()
    reference_loss = torch.tensor([v["reference-loss"] for v in self.losses]).mean().item()

    self.current_loss_dict = {
            "loss.training-loss": loss,
            "rel-metric.relative-loss": reference_loss - current_loss,
            "rel-metric.current-loss": current_loss,
            "rel-metric.reference-loss": reference_loss,
      }

    self.cfg.mlops.log_metric_dictionary(self.current_loss_dict,self.current_epoch)
        
  def compute_loss_value(self, x, y):
    loss = 0
    for loss_inst, loss_w in zip(self.cfg.loss_instances, self.cfg.loss_weight):
      loss += loss_w * loss_inst(x,y)
    return loss

  def training_step(self, batch):
    target_index = 0
    if self.current_epoch < self.cfg.epochs_random:
      target_index = np.random.randint(4)
    raw_8f_images = batch[target_index].cuda()
    synthesize_32f_images = batch[4].cuda()
    raw_8f_paths = batch[5]
    denoised_32f_images = self.model(raw_8f_images)

    self.default_size = raw_8f_images.shape[-1]
    loss = self.compute_loss_value(denoised_32f_images, synthesize_32f_images)

    ## Ground Truth - Denoised
    intensity_loss = (synthesize_32f_images - denoised_32f_images).abs().mean(dim = (1,2,3)).mean().detach()
    ## Ground Truth - Noisy
    reference_loss = (synthesize_32f_images - raw_8f_images).abs().mean(dim = (1,2,3)).mean().detach()

    loss_dict = {
        "loss" : loss,
        "current-loss" : intensity_loss,
        "reference-loss" : reference_loss
    }
    return loss_dict
  
  def save(self, is_last = False):
    def convert_and_save_torch_model(model, prefix):
      clone_model = copy.deepcopy(self.model)
      sample_input = torch.zeros(
                (1, 1, self.default_size, self.default_size)).cuda()
      clone_model = clone_model.float()
      sample_input = sample_input.float()

      #output_name = os.path.splitext(os.path.basename(input_path))[0]
      os.makedirs("models", exist_ok=True)

      output_name = "models/" + prefix + self.recipe_name
      output_name += "-GPU"
      output_name += "-FULL"
      output_name += ".pt"

      traced_model = torch.jit.trace(clone_model, sample_input)
      torch.jit.save(traced_model, output_name)
      self.cfg.mlops.save_artifact(output_name, "results")

      ## TODO set normal parmas from network definition
      denoiser_configuration = {
                "model": {
                    "path": os.path.basename(output_name)
                },
                "input-data": {
                    "data-type": "half",
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
            }
      with open("models/AIMSDenoiser.json", "w", encoding="utf8") as f:
        json.dump(denoiser_configuration, f, indent=4)

      self.cfg.mlops.save_artifact("models/AIMSDenoiser.json", "results")
    
    state_dict = {
        'epoch' : self.current_epoch,
        'state_dict' : self.model.state_dict(),
        'optimizer' : self.cfg.optimizer.state_dict()
    }

    current_metric = self.current_metric_dict["metric.dsn-vs-syn.rsq"]
    if self.best_metric is None:
      self.best_metric = current_metric

    if self.best_metric <= current_metric:
      self.best_metric = current_metric
      torch.save(state_dict, "best.pth")
      convert_and_save_torch_model(self.model, "best-")
      self.cfg.mlops.save_artifact("best.pth", "results")
      self.rendering_samples()
      self.cfg.mlops.save_models()

    if is_last:
      torch.save(state_dict, "last.pth")
      convert_and_save_torch_model(self.model, "last-")
      self.cfg.mlops.save_artifact("last.pth", "results")
      self.rendering_samples()
      self.cfg.mlops.save_models()
      
  def rendering_samples(self):
    if not os.path.exists("./samples"):
      os.mkdir("./samples")

    with torch.no_grad():
      dataset = self.eval_data_loader.dataset
      for batch in self.eval_data_loader:
        raw_8f_images = batch[0].cuda()
        synthesize_32f_images = batch[4].cuda()
        raw_8f_paths = batch[5]
        denoised_32_images = self.model(raw_8f_images)
        dataset.save_tensor(
                    "samples/sample-%04d-raw-8f.png" % (self.current_epoch),
                    raw_8f_images[0])
        dataset.save_tensor(
                    "samples/sample-%04d-dsn-32f.png" % (self.current_epoch),
                    denoised_32_images[0])
        dataset.save_tensor(
                    "samples/sample-%04d-syn-32f.png" % (self.current_epoch),
                    synthesize_32f_images[0])
        self.cfg.mlops.save_artifacts("samples", "samples")
        break





