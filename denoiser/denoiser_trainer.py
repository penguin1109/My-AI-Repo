import os
import json
import copy
import tqdm
import hydra
import torch
import mlflow
import shutil

import numpy as np

from code.aims_trainer import AIMSTrainer
from code.models import build_model
from code.optimizer import build_optimizer
from code.scheduler import build_scheduler
from code.loss import build_loss_instance
from gli_aims.tools.mlops.aims_mlops import ManagedMLFlow


class AIMSDenoiserTrainer(AIMSTrainer):

    def __init__(self, recipe_name, train_configuration,
                 mlops_configuration, model_configuration):
        super().__init__()

        self.train_configuration = train_configuration
        self.mlops_configuration = mlops_configuration
        self.model_configuration = model_configuration
        self.recipe_name = recipe_name

        self.current_loss_dict = None
        self.current_metric_dict = None

    def configuration(self):
        self.model_name = self.model_configuration["backbone"]["name"]

        self.model = build_model(self.model_configuration)
        self.optimizer = build_optimizer(self.train_configuration["optimizer"],
                                         self.model.parameters())
        self.lr_scheduler = build_scheduler(
            self.train_configuration["scheduler"], self.optimizer)

        strategy = self.train_configuration["strategy"]
        self.epochs_random = strategy[0]["epoch"]
        self.epochs_zero = strategy[1]["epoch"]
        self.epochs = self.epochs_random + self.epochs_zero

        #self.save_path = 'saved_checkpoints/' + '{}_{}'.format(
        #    self.model_name, self.datestr())
        #self.save_name = '{}_{}'.format(self.model_name, self.datestr())
        self.register_loss_functions(self.train_configuration["loss_functions"])

        self.mlops = ManagedMLFlow(
            experiment_name=self.mlops_configuration["experiment_name"],
            run_name=self.mlops_configuration["run_name"],
            user_name=self.mlops_configuration["user_name"],
            tracking_uri=self.mlops_configuration["tracking_uri"])

        original_path = hydra.utils.get_original_cwd()
        self.mlops.save_artifacts(os.path.join(original_path, "code"), "code")
        self.mlops.save_artifact(
            os.path.join(original_path, "aims_denoiser_trainer_entry.py"))

        self.mlops.log_param_dictionary(self.train_configuration)

        self.best_metric = None

        self.copy_inline_32f_datasets()

        ### LOAD ROUTINE ###
        #self.model.load_state_dict(torch.load("/project/workSpace/aims-yrepo/devspaces/aims-denoiser/task01/gausslabs/experimental/teams/aims/denoiser-test/last.pth")["state_dict"])

    def register_loss_functions(self, criterion_config):
        self.loss_instances = []
        self.loss_lambdas = []

        for criteria in criterion_config:
            self.loss_instances.append(build_loss_instance(criteria["type"]))
            self.loss_lambdas.append(criteria["lambda"])

    def compute_loss_value(self, x, y):
        loss = 0
        for loss_instance, loss_lambda in zip(self.loss_instances,
                                              self.loss_lambdas):
            loss += loss_lambda * loss_instance(x, y)
        return loss

    def training_epoch_start(self):
        self.model.train()
        self.losses = []

    ## TODO: max or min
    def update_current_dict(self, target_dict, current_dict):
        if target_dict is None:
            target_dict = current_dict
            target_dict.update(
                {"best." + key: value for key, value in current_dict.items()})
        else:
            target_dict.update(current_dict)
            for key, value in current_dict.items():
                if target_dict["best." + key] <= current_dict[key]:
                    target_dict["best." + key] = current_dict[key]
        return target_dict

    def training_epoch_end(self):
        loss = torch.tensor([v["loss"] for v in self.losses]).mean().item()
        current_loss = torch.tensor([v["current-loss"] for v in self.losses
                                    ]).mean().item()
        reference_loss = torch.tensor(
            [v["reference-loss"] for v in self.losses]).mean().item()

        self.current_loss_dict = {
            "loss.training-loss": loss,
            "rel-metric.relative-loss": reference_loss - current_loss,
            "rel-metric.current-loss": current_loss,
            "rel-metric.reference-loss": reference_loss,
        }

        self.mlops.log_metric_dictionary(self.current_loss_dict,
                                         self.current_epoch)

    def training_step_start(self):
        pass

    def training_step(self, batch):
        target_index = 0
        if self.current_epoch < self.epochs_random:
            target_index = np.random.randint(4)

        raw_8f_images = batch[target_index].cuda()
        synthesize_32f_images = batch[4].cuda()
        raw_8f_paths = batch[5]
        denoised_32f_images = self.model(raw_8f_images)

        self.default_size = raw_8f_images.shape[-1]

        loss = self.compute_loss_value(denoised_32f_images,
                                       synthesize_32f_images)

        # for monitoring progress
        intensity_loss = (synthesize_32f_images -
                          denoised_32f_images).abs().mean(
                              dim=(1, 2, 3)).mean().detach()
        reference_loss = (synthesize_32f_images -
                          raw_8f_images).abs().mean(dim=(1, 2,
                                                         3)).mean().detach()

        loss_dict = {
            "loss": loss,
            "current-loss": intensity_loss,
            "reference-loss": reference_loss,
        }
        return loss_dict

    def training_step_end(self, loss_dict, epoch_tqdm):
        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        self.losses.append(loss_dict)

        epoch_tqdm.set_description(
            "[Train][Epoch %03d][Iter %04d]"
            " Loss: %.5f | Base-Loss: %.5f | Ref-Loss: %.5f | Diff: %.5f" %
            (self.current_epoch, self.current_step, loss_dict["loss"],
             loss_dict["current-loss"], loss_dict["reference-loss"],
             loss_dict["reference-loss"] - loss_dict["current-loss"]))

    def save(self, is_last=False):

        ## TODO: Modularize this as well-defined class
        def convert_and_save_torch_model(model, prefix):
            clone_model = copy.deepcopy(self.model)
            sample_input = torch.zeros(
                (1, 1, self.default_size, self.default_size)).cuda()
            for half in [False, True]:
                clone_model = clone_model.half() if half else clone_model.float(
                )
                sample_input = sample_input.half(
                ) if half else sample_input.float()

                #output_name = os.path.splitext(os.path.basename(input_path))[0]
                os.makedirs("models", exist_ok=True)

                output_name = "models/" + prefix + self.recipe_name
                output_name += "-GPU"
                output_name += "-HALF" if half else "-FULL"
                output_name += ".pt"

                traced_model = torch.jit.trace(clone_model, sample_input)
                torch.jit.save(traced_model, output_name)
                self.mlops.save_artifact(output_name, "results")

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

            self.mlops.save_artifact("models/AIMSDenoiser.json", "results")

        state_dict = {
            'epoch': self.current_epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        current_metric = self.current_metric_dict["metric.dsn-vs-syn.rsq"]
        if self.best_metric is None:
            self.best_metric = current_metric

        if self.best_metric <= current_metric:
            self.best_metric = current_metric
            torch.save(state_dict, "best.pth")
            convert_and_save_torch_model(self.model, "best-")
            self.mlops.save_artifact("best.pth", "results")
            self.rendering_samples()
            self.mlops.save_models()

        if is_last:
            torch.save(state_dict, "last.pth")
            convert_and_save_torch_model(self.model, "last-")
            self.mlops.save_artifact("last.pth", "results")
            self.rendering_samples()
            self.mlops.save_models()

        #self.mlops.save_model(self.model,
        #    "aims-denoiser-%04d" % (self.current_epoch))

    def rendering_samples(self):
        # TODO: be aware of this, 777 is dangerous
        #os.makedirs("./samples", mode=777, exist_ok=True)
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
                # TODO: should be improved to reduce the redundant copy
                self.mlops.save_artifacts("samples", "samples")
                break

    def evaluation(self):
        self.model.eval()

        self.generate_datasets()
        stats = self.evaluator.run()

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

        self.current_metric_dict = self.update_current_dict(
            self.current_metric_dict, current_metric_dict)
        self.mlops.log_metric_dictionary(self.current_metric_dict,
                                         self.current_epoch)
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
                    os.path.join(output_path,
                                 "S24_M%04d-02MS.png" % (image_index)),
                    image_tensors[k])

        with torch.no_grad():
            for batch in tqdm.tqdm(self.eval_data_loader):
                raw_8f_images = batch[0]
                synthesize_32f_images = batch[4].cuda()
                raw_8f_paths = batch[5]
                denoised_32_images = self.model(batch[0].cuda())

                save_images(raw_8f_images, raw_8f_paths, root_path,
                            "outputs-raw")
                save_images(denoised_32_images, raw_8f_paths, root_path,
                            "outputs-dsn")
                save_images(synthesize_32f_images, raw_8f_paths, root_path,
                            "outputs-syn")
