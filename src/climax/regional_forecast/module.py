# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# credits: https://github.com/ashleve/lightning-hydra-template/blob/main/src/models/mnist_module.py
from typing import Any

import torch
from pytorch_lightning import LightningModule
from torchvision.transforms import transforms

from climax.regional_forecast.arch import RegionalClimaX
from climax.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from climax.utils.metrics import (
    lat_weighted_acc,
    lat_weighted_mse,
    lat_weighted_mse_val,
    lat_weighted_rmse,
    lat_weighted_mae,
    lat_weighted_mae_val,
    pressure_lat_weighted_mse,
    freq_lat_weighted_mae,
    freq_lat_weighted_mse,
    freq_lat_weighted_mae_val,
    freq_lat_weighted_mse_val,
    auxillary_loss,
    auxillary_loss_val
)
from climax.utils.pos_embed import interpolate_pos_embed
from peft import LoraConfig, get_peft_model
from typing import Union, List, Pattern
from climax.utils.data_utils import CHEMICAL_VARS
import numpy as np

BATCH_NUM = 0

class RegionalForecastModule(LightningModule):
    """Lightning module for regional forecasting with the ClimaX model.

    Args:
        net (ClimaX): ClimaX model.
        pretrained_path (str, optional): Path to pre-trained checkpoint.
        lr (float, optional): Learning rate.
        beta_1 (float, optional): Beta 1 for AdamW.
        beta_2 (float, optional): Beta 2 for AdamW.
        weight_decay (float, optional): Weight decay for AdamW.
        warmup_epochs (int, optional): Number of warmup epochs.
        max_epochs (int, optional): Number of total epochs.
        warmup_start_lr (float, optional): Starting learning rate for warmup.
        eta_min (float, optional): Minimum learning rate.
    """

    def __init__(
        self,
        net: RegionalClimaX,
        pretrained_path: str = "",
        lr: float = 5e-4,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 10000,
        max_epochs: int = 200000,
        warmup_start_lr: float = 1e-8,
        eta_min: float = 1e-8,
        peft_lora: bool = False,
        peft_method: str = "lora",
        r: int = 8,
        lora_alpha: int = 16,
        target_modules: Union[List[str], str, Pattern[str]] = r".*\.attn.qkv",
        lora_dropout: float = 0.1,
        bias: Union[str, None] = "none",
        modules_to_save: Union[List[str], str, Pattern[str]] = ["head"],
        multi_step_rollout: bool = False,
        k: int = 4,
        loss_method: str = "pressure_lat_weighted_mse",
        visualize_pred: bool = False,
        visualize_period: List[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net"])
        self.net = net
        self.multi_step_rollout = multi_step_rollout
        self.k = k
        self.visualize_pred = visualize_pred
        self.visualize_period = visualize_period

        # we assign the loss function based on the loss_method
        if loss_method == "pressure_lat_weighted_mse":
            self.loss_fn = pressure_lat_weighted_mse
        elif loss_method == "lat_weighted_mse":
            self.loss_fn = lat_weighted_mse
            self.val_loss_fn = lat_weighted_mse_val
        elif loss_method == "freq_lat_weighted_mse":
            self.loss_fn = freq_lat_weighted_mse
            self.val_loss_fn = freq_lat_weighted_mse_val
        elif loss_method == "freq_lat_weighted_mae":
            self.loss_fn = freq_lat_weighted_mae
            self.val_loss_fn = freq_lat_weighted_mae_val
        elif loss_method == "lat_weighted_mae":
            self.loss_fn = lat_weighted_mae
            self.val_loss_fn = lat_weighted_mae_val
        elif loss_method == "auxillary_loss":
            self.loss_fn = auxillary_loss
            self.val_loss_fn = auxillary_loss_val


        
        if len(pretrained_path) > 0:
            self.load_pretrained_weights(pretrained_path)

        # loading lora config and lora model 
        if peft_lora:
            use_dora = peft_method == "dora"
            config = LoraConfig(
                use_dora=use_dora,
                r=r,
                init_lora_weights='pissa_niter_4',
                lora_alpha=lora_alpha,
                target_modules=target_modules,
                lora_dropout=lora_dropout,
                bias=bias,
                modules_to_save=modules_to_save,
            )
            self.net = get_peft_model(self.net, config)
    

    def load_pretrained_weights(self, pretrained_path):
        if pretrained_path.startswith("http"):
            checkpoint = torch.hub.load_state_dict_from_url(pretrained_path)
        else:
            checkpoint = torch.load(pretrained_path, map_location=torch.device("cpu"))

        print("Loading pre-trained checkpoint from: %s" % pretrained_path)
        checkpoint_model = checkpoint["state_dict"]
        # interpolate positional embedding
        interpolate_pos_embed(self.net, checkpoint_model, new_size=self.net.img_size)

        
        # interpolate ViT blocks if the number of blocks in the checkpoint model is less than the current model
        ckpt_blocks = [(".").join(k.split(".")[0:3]) for k in checkpoint_model.keys() if "blocks" in k]
        ckpt_blocks = list(set(ckpt_blocks))
        ckpt_blocks.sort()
        if len(ckpt_blocks) != len(self.net.blocks):
            print('interpolating blocks')
            print('blocks in checkpoint', len(ckpt_blocks))
            print('blocks in model', len(self.net.blocks))

            # we use the last block in the checkpoint model to interpolate the blocks in the current model
            last_block = ckpt_blocks[-1]

            block_num = int(last_block.split(".")[-1])

            block_vals = {}
            for k in checkpoint_model.keys():
                if last_block in k:
                    block_vals[k] = checkpoint_model[k]

            while block_num < len(self.net.blocks) - 1:
                block_num += 1
                block_key = f"net.blocks.{block_num}"
                for k in block_vals.keys():
                    checkpoint_model[k.replace(last_block, block_key)] = block_vals[k]
                

        state_dict = self.state_dict()
        if self.net.parallel_patch_embed:
            if "token_embeds.proj_weights" not in checkpoint_model.keys():
                raise ValueError(
                    "Pretrained checkpoint does not have token_embeds.proj_weights for parallel processing. Please convert the checkpoints first or disable parallel patch_embed tokenization."
                )

        # checkpoint_keys = list(checkpoint_model.keys())
        for k in list(checkpoint_model.keys()):
            if "channel" in k:
                checkpoint_model[k.replace("channel", "var")] = checkpoint_model[k]
                del checkpoint_model[k]
        for k in list(checkpoint_model.keys()):
            if k not in state_dict.keys() or checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # load pre-trained model
        msg = self.load_state_dict(checkpoint_model, strict=False)
        print(msg)

    def set_out_vars(self, out_vars):
        self.out_vars = out_vars

    def set_denormalization(self, mean, std):
        self.denormalization = transforms.Normalize(mean, std)

    def inverse_transform(self):
        def inverse_transform(inp):
            for batch in range(inp.shape[0]):
                for var in self.out_vars:
                    if var in CHEMICAL_VARS:
                        # inp[self.out_vars.index(var)] = torch.exp(inp[batch, self.out_vars.index(var)])
                        tmp = (np.log(1e-4) * inp[batch, self.out_vars.index(var)]) + np.log(1e-4)
                        inp[batch, self.out_vars.index(var)] = torch.exp(tmp)
            return inp
        
        return transforms.Lambda(inverse_transform)



    def set_lat_lon(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def set_pred_range(self, r):
        self.pred_range = r

    def set_val_clim(self, clim):
        self.val_clim = clim

    def set_test_clim(self, clim):
        self.test_clim = clim

    def get_patch_size(self):
        return self.net.patch_size

    def training_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch
        if self.multi_step_rollout:
            assert self.k > 1, "k should be greater than 1 for multi-step rollout"
            y = y[:, 0, :, :, :] # TODO: Make sure the index is correct

            loss_dict, preds_weather, preds_chem = self.net.forward(
                x, y, lead_times, variables, out_variables, [self.loss_fn], lat=self.lat, region_info=region_info
            )
            # we try to follow stormer's multi step roll out. We average the losses from k steps, and take the output from the last step as input to the next step
            for i in range(self.k-1):
                x = torch.cat([preds_weather, preds_chem], dim=1)
                y = y[:, i+1, :, :, :]
                loss_dict_temp, preds_weather, preds_chem = self.net.forward(
                    x, y, lead_times, variables, out_variables, [self.loss_fn], lat=self.lat, region_info=region_info
                )
                for key in loss_dict.keys():
                    loss_dict[key] += loss_dict_temp[key]
            for key in loss_dict.keys():
                loss_dict[key] /= self.k

        else:
            loss_dict, _, _ = self.net.forward(
                x, y, lead_times, variables, out_variables, [self.loss_fn], lat=self.lat, region_info=region_info
            )
            # loss_dict = loss_dict
        
        for var in loss_dict.keys():
            self.log(
                "train/" + var,
                loss_dict[var],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )
        loss = loss_dict["loss"]
 

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=transforms.Compose([self.inverse_transform(), self.denormalization]),
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc, self.val_loss_fn],
            lat=self.lat,
            clim=self.val_clim,
            log_postfix=log_postfix,
            region_info=region_info,
        )

        loss_dict = {}
        # print(all_loss_dicts)
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]

        for var in loss_dict.keys():
            self.log(
                "val/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )


        return loss_dict

    def test_step(self, batch: Any, batch_idx: int):
        x, y, lead_times, variables, out_variables, region_info = batch
        global BATCH_NUM
        # we used the below lines when we wanted to test the model with the CAMS forecasts which was daily and not hourly
        # BATCH_NUM += 1
        # if BATCH_NUM >= 8760: 
        #     return
        # if BATCH_NUM % 24 != 0:
        #     return

        if self.pred_range < 24:
            log_postfix = f"{self.pred_range}_hours"
        else:
            days = int(self.pred_range / 24)
            log_postfix = f"{days}_days"

        all_loss_dicts = self.net.evaluate(
            x,
            y,
            lead_times,
            variables,
            out_variables,
            transform=transforms.Compose([self.inverse_transform(), self.denormalization]),
            metrics=[lat_weighted_mse_val, lat_weighted_rmse, lat_weighted_acc],
            lat=self.lat,
            clim=self.test_clim,
            log_postfix=log_postfix,
            region_info=region_info,
            visualize_pred=self.visualize_pred,
            lon = self.lon if self.visualize_pred else None,
            visualize_period = self.visualize_period
        )

        loss_dict = {}
        for d in all_loss_dicts:
            for k in d.keys():
                loss_dict[k] = d[k]
        for var in loss_dict.keys():
            self.log(
                "test/" + var,
                loss_dict[var],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
            )


        return loss_dict

    def configure_optimizers(self):
        decay = []
        no_decay = []
        for name, m in self.named_parameters():
            if "var_embed" in name or "pos_embed" in name or "time_pos_embed" in name:
                no_decay.append(m)
            else:
                decay.append(m)

        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": self.hparams.weight_decay,
                },
                {
                    "params": no_decay,
                    "lr": self.hparams.lr,
                    "betas": (self.hparams.beta_1, self.hparams.beta_2),
                    "weight_decay": 0,
                },
            ]
        )

        lr_scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            self.hparams.warmup_epochs,
            self.hparams.max_epochs,
            self.hparams.warmup_start_lr,
            self.hparams.eta_min,
        )
        scheduler = {"scheduler": lr_scheduler, "interval": "step", "frequency": 1}

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
