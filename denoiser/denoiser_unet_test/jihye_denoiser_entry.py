## denoiser_unet_test/jihye_trainer_entry.py

import os
os.environ["AWS_ACCESS_KEY_ID"] = "yonghyun"
os.environ["AWS_SECRET_ACCESS_KEY"] = "gausslabs"
os.environ["MLFLOW_S3_IGNORE_TLS"] = "1"
os.environ[
    "MLFLOW_S3_ENDPOINT_URL"] = "http://aims-object-storage-basic-dev.api.sddc-gl.skhynix.com:80"

import sys, json, hydra
from hydra.core.hydra_config import HydraConfig
import torch.utils.data as data
from omegaconf import DictConfig, OmegaConf

from gli_aims.tools.aims_data_interface import AIMSDataInterface_Gauss as DataInterface
from gli_aims.tools.aims_asset_interface import AIMSAssetInterface as AssetInterface

from denoiser_test.code.datasets.aims_multi_frame_dataset import AIMSMultiFrameDataset
from denoiser_test.code.evalutaion.aims_denoiser_evaluator import AIMSDenoiserEvaluator
from jihye_denoiser_trainer import DenoiserTrainer

class trainer_entry():
  def __init__(self, cfg_instance):
    self.cfg = cfg_instance
  
  def __call__(self, config : DictConfig) -> None:
    gpu_id = 0
    if not OmegaConf.is_missing(HydraConfig.get().job, "num"):
      gpu_id = HydraConfig.get().job.num % 4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    mlops_cfg = OmegaConf.to_container(OmegaConf.create(self.cfg.mlops_pipe))
    train_cfg = OmegaConf.to_container(OmegaConf.create(self.cfg.train_pipe))
    model_cfg = OmegaConf.to_container(OmegaConf.create(self.cfg.model_cfg))
    eval_cfg = OmegaConf.to_container(OmegaConf.create(self.cfg.eval_pipe))

    interface_cfg_pth = self.cfg.data_cfg['aims_interface_configuration']
    target_recipe = self.cfg.recipe
    batch_size = train_cfg['batch_size']

    with open(interface_cfg_pth, encoding = 'utf-8') as f:
      json_data = json.load(f)
    
    aims_trainer_configuration = os.path.join(
            "/project/workSpace/gauss-nas/dev/aims/aims-recipe/app-assets/recipes",
            "AIMSRecipe_%s/" % (target_recipe), "aims-trainer-denoiser.json")
    
    with open(aims_trainer_configuration, encoding = 'utf-8') as f:
      aims_trainer_config = json.load(f) ## 각 recipe에 따라 다른 denoiser training config
    
    data_interface = DataInterface(json_data)
    data_interface.load_db(
            aims_trainer_config["train-scheme"]["dataset"]["input-data"])

    asset_interface = AssetInterface(json_data)
    asset_interface.read_structure()

    filtered_wafers = data_interface.query_wafers(
            f"(cd_recipe_name == \"{target_recipe}\")")
    
    filtered_wafers = data_interface.convert_data_to_aims_input(
            filtered_wafers)
    assets = asset_interface.convert_data_to_aims_input(
            asset_interface.assets)

    amf_dataset = AIMSMultiFrameDataset(
            assets,
            filtered_wafers,
            parameter_mapping=aims_trainer_config["train-scheme"]["dataset"]["parameter-mapping"],
            mode="train")
    
    amf_data_loader = data.DataLoader(amf_dataset, batch_size, shuffle = True)
    eval_amf_dataset = AIMSMultiFrameDataset(
            assets,
            filtered_wafers,
            parameter_mapping=aims_trainer_config["train-scheme"]["dataset"]["parameter-mapping"],
            mode="eval")

    eval_amf_data_loader = data.DataLoader(eval_amf_dataset,
                                               batch_size,
                                               shuffle=False)

    evaluator = AIMSDenoiserEvaluator(json_data, target_recipe, 
                                          aims_trainer_config, eval_cfg)
    trainer = DenoiserTrainer(self.cfg) ## trainer made
    ret = trainer.run(amf_data_loader, eval_amf_data_loader, eval_cfg, evaluator)
    trainer.cfg.mlops.end_run()

    return ret

 
