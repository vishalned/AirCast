# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from scipy import stats
from climax.utils.data_utils import CHEMICAL_VARS
import json


BINS = json.load(open("./aircast_data/bins.json", "r"))
COUNTS = json.load(open("./aircast_data/counts.json", "r"))

def mse(pred, y, vars, lat=None, mask=None):
    """Mean squared error

    Args:
        pred: [B, L, V*p*p]
        y: [B, V, H, W]
        vars: list of variable names
    """

    loss = (pred - y) ** 2

    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (loss[:, i] * mask).sum() / mask.sum()
            else:
                loss_dict[var] = loss[:, i].mean()

    if mask is not None:
        loss_dict["loss"] = (loss.mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = loss.mean(dim=1).mean()

    return loss_dict

def pressure_lat_weighted_mse(pred, y, vars, lat, mask=None):
    """
        Pressure and Lat weighted MSE
    """

    error = (pred - y) ** 2  # [N, C, H, W]
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    pressure_levels = ['50', '250', '500', '600', '700', '850', '925']
    for var in vars:
        if var == "2m_temperature" or var in CHEMICAL_VARS and '_' not in var:
            # we only want the 2m temperature and the surface chem variables. Any variable with an underscore is a
            # pressure level variable
            w_pres = 1.0
        else:
            matching_string = [string for string in pressure_levels if string in var]
            if len(matching_string) == 0:
                w_pres = 0.1
            else:
                w_pres = int(matching_string[0]) / 1000

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_pres * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_pres * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_pres * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_pres * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict



def lat_weighted_mse(pred, y, vars, lat, mask=None):
    """Latitude weighted mean squared error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict
def lat_weighted_mae(pred, y, vars, lat, mask=None):
    """Latitude weighted mean absolute error

    Allows to weight the loss by the cosine of the latitude to account for gridding differences at equator vs. poles.

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = torch.abs(pred - y)  # [N, C, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict

def lat_weighted_mae_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean abs error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = torch.abs(pred - y)  # [B, V, H, W]
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mae_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_mse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_mse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def freq_lat_weighted_mse(pred, y, vars, lat, mask=None):
    """
    To the lat weighted MSE, we add an additional frequency weighting based on which bin a pixel falls into.

    TESTING --- For testing purposes, the frequency is the bins itself. hence we just use the bins for now. This is because in pm2.5, pm10, pm1 the first bin has the highest 
    frequency. so we assign that the lowest weight. this is not true for other variables, but for now we only have those variables. 

    """
    
    error = (pred - y) ** 2  # [N, C, H, W]
    tmp_pred = pred.clone()
    tmp_pred = tmp_pred.detach().cpu().numpy()

    # we only apply the freq weighted loss to the chemical variables
    for var in vars:
        if var in CHEMICAL_VARS:
            bins_ind = np.digitize(tmp_pred[:, vars.index(var)], BINS[var], right=True) - 1
            bins_ind[bins_ind == len(BINS[var]) - 1] = len(BINS[var]) - 2 # incase anything is more than the last bin
            w_freq = np.zeros(len(bins_ind))
              

            # we need a bin_ind_count to get the count of the bin index. the index in COUNTS is the bin index, we need the value
            bin_ind_count = [COUNTS[var][i] for i in bins_ind.flatten()]
            bin_ind_count = np.array(bin_ind_count).reshape(bins_ind.shape)

            mask_zero = bin_ind_count == 0
            mask_one = bin_ind_count == 1

            w_freq = np.where(mask_zero, 0,
                    np.where(mask_one, 1/np.log(bin_ind_count+0.2), 1/np.log(bin_ind_count)))
                    
            w_freq = torch.from_numpy(w_freq).to(dtype=error.dtype, device=error.device)
            error[:, vars.index(var)] = error[:, vars.index(var)] * w_freq

    

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()

    return loss_dict


def freq_lat_weighted_mae(pred, y, vars, lat, mask=None):
    """
    To the lat weighted MAE, we add an additional frequency weighting based on which bin a pixel falls into.

    TESTING --- For testing purposes, the frequency is the bins itself. hence we just use the bins for now. This is because in pm2.5, pm10, pm1 the first bin has the highest 
    frequency. so we assign that the lowest weight. this is not true for other variables, but for now we only have those variables. 

    """


   
    error = torch.abs(pred - y)  # [N, C, H, W]
    tmp_pred = pred.clone()
    tmp_pred = tmp_pred.detach().cpu().numpy()

    # we only apply the freq weighted loss to the chemical variables
    for var in vars:
        if var in CHEMICAL_VARS:
            bins_ind = np.digitize(tmp_pred[:, vars.index(var)], BINS[var], right=True) - 1
            bins_ind[bins_ind == len(BINS[var]) - 1] = len(BINS[var]) - 2 # incase anything is more than the last bin

            w_freq = np.zeros(len(bins_ind))
            # we need a bin_ind_count to get the count of the bin index. the index in COUNTS is the bin index, we need the value
            bin_ind_count = [COUNTS[var][i] for i in bins_ind.flatten()]
            bin_ind_count = np.array(bin_ind_count).reshape(bins_ind.shape)


            # we try a new method based on the Class-Balanced Loss Based on Effective Number of Samples paper.
            # we first calculate the effective number of samples based on the formula E = 1-Beta ^ C / (1-Beta), where C is the number of classes (bins in this case)
            # we then calculate the weight for each bin based on the formula w = 1 / E
            beta = 0.8 # 0.8 is exp number 4

            # if the num of bins is 0, then the weight is 0
            mask_zero = bin_ind_count == 0
            E_num = (1 - beta ** bin_ind_count) / (1 - beta)
            w_freq = np.where(mask_zero, 0, 1 / E_num)


            
            # based on the method written in the paper
            # mask_zero = bin_ind_count == 0
            # mask_one = bin_ind_count == 1 

            # w_freq = np.where(mask_zero, 0,
            #         np.where(mask_one, 1/np.log(bin_ind_count+0.2), 1/np.log(bin_ind_count)))
            
            w_freq = torch.from_numpy(w_freq).to(dtype=error.dtype, device=error.device)
            error[:, vars.index(var)] = error[:, vars.index(var)] * w_freq




    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            if mask is not None:
                loss_dict[var] = (error[:, i] * w_lat * mask).sum() / mask.sum()
            else:
                loss_dict[var] = (error[:, i] * w_lat).mean()

    if mask is not None:
        loss_dict["loss"] = ((error * w_lat.unsqueeze(1)).mean(dim=1) * mask).sum() / mask.sum()
    else:
        loss_dict["loss"] = (error * w_lat.unsqueeze(1)).mean(dim=1).mean()
    # exit()
    return loss_dict

def freq_lat_weighted_mse_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean squared error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = (pred - y) ** 2  # [B, V, H, W]
    tmp_pred = pred.clone()
    tmp_pred = tmp_pred.detach().cpu().numpy()

    # we only apply the freq weighted loss to the chemical variables
    for var in vars:
        if var in CHEMICAL_VARS:
            bins_ind = np.digitize(tmp_pred[:, vars.index(var)], BINS[var], right=True) - 1
            bins_ind[bins_ind == len(BINS[var]) - 1] = len(BINS[var]) - 2 # incase anything is more than the last bin
            w_freq = np.zeros(len(bins_ind))
            # we need a bin_ind_count to get the count of the bin index. the index in COUNTS is the bin index, we need the value
            bin_ind_count = [COUNTS[var][i] for i in bins_ind.flatten()]
            bin_ind_count = np.array(bin_ind_count).reshape(bins_ind.shape)

            mask_zero = bin_ind_count == 0
            mask_one = bin_ind_count == 1

            w_freq = np.where(mask_zero, 0,
                    np.where(mask_one, 1/np.log(bin_ind_count+0.2), 1/np.log(bin_ind_count)))
            w_freq = torch.from_numpy(w_freq).to(dtype=error.dtype, device=error.device)
            error[:, vars.index(var)] = error[:, vars.index(var)] * w_freq

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_fmse_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_fmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

def freq_lat_weighted_mae_val(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted mean abs error
    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    error = torch.abs(pred - y)  # [B, V, H, W]
    tmp_pred = pred.clone()
    tmp_pred = tmp_pred.detach().cpu().numpy()

    # we only apply the freq weighted loss to the chemical variables
    for var in vars:
        if var in CHEMICAL_VARS:
            bins_ind = np.digitize(tmp_pred[:, vars.index(var)], BINS[var], right=True) - 1
            bins_ind[bins_ind == len(BINS[var]) - 1] = len(BINS[var]) - 2 # incase anything is more than the last bin
            w_freq = np.zeros(len(bins_ind))
            
            # we need a bin_ind_count to get the count of the bin index. the index in COUNTS is the bin index, we need the value
            bin_ind_count = [COUNTS[var][i] for i in bins_ind.flatten()]
            bin_ind_count = np.array(bin_ind_count).reshape(bins_ind.shape)


            beta = 0.8

            # if the num of bins is 0, then the weight is 0
            mask_zero = bin_ind_count == 0
            E_num = (1 - beta ** bin_ind_count) / (1 - beta)
            w_freq = np.where(mask_zero, 0, 1 / E_num)
            
            w_freq = torch.from_numpy(w_freq).to(dtype=error.dtype, device=error.device)
            error[:, vars.index(var)] = error[:, vars.index(var)] * w_freq

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)  # (1, H, 1)
    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_fmae_{var}_{log_postfix}"] = (error[:, i] * w_lat).mean()

    loss_dict["w_fmae"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict

AUX_WEIGHT1 = 0.75
AUX_WEIGHT2 = 0.25
def auxillary_loss(pred, y, vars, lat, mask=None):
    """
    Auxillary loss is the sum of the 2 kinds of losses
    """
    loss_dict = {}
    loss_dict1 = lat_weighted_mae(pred, y, vars, lat, mask)
    loss_dict2 = freq_lat_weighted_mae(pred, y, vars, lat, mask) # make the MAE
    loss_dict['loss'] = AUX_WEIGHT1*loss_dict1['loss'] + AUX_WEIGHT2*loss_dict2['loss']
    for key in loss_dict1.keys():
        if key != "loss":
            loss_dict[key] = loss_dict1[key]
    for key in loss_dict2.keys():
        if key != "loss":
            loss_dict[key] = loss_dict2[key]

    return loss_dict

def auxillary_loss_val(pred, y, transform, vars, lat, clim, log_postfix):
    """
    Auxillary loss is the sum of the 2 kinds of losses
    """
    loss_dict = {}
    loss_dict1 = lat_weighted_mae_val(pred, y, transform, vars, lat, clim, log_postfix)
    loss_dict2 = freq_lat_weighted_mae_val(pred, y, transform, vars, lat, clim, log_postfix)
    loss_dict['aux_loss'] = AUX_WEIGHT1*loss_dict1['w_mae'] + AUX_WEIGHT2*loss_dict2['w_fmae']
    for key in loss_dict1.keys():
        if key != "w_mae":
            loss_dict[key] = loss_dict1[key]
    for key in loss_dict2.keys():
        if key != "w_fmae":
            loss_dict[key] = loss_dict2[key]

    return loss_dict

def lat_weighted_rmse(pred, y, transform, vars, lat, clim, log_postfix):
    """Latitude weighted root mean squared error

    Args:
        y: [B, V, H, W]
        pred: [B, V, H, W]
        vars: list of variable names
        lat: H
    """

    # pred = transform(pred)
    # y = transform(y)
    error = (pred - y) ** 2  # [B, V, H, W]

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=error.dtype, device=error.device)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            loss_dict[f"w_rmse_{var}_{log_postfix}"] = torch.mean(
                torch.sqrt(torch.mean(error[:, i] * w_lat, dim=(-2, -1)))
            )

    loss_dict["w_rmse"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_acc(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    # pred = transform(pred)
    # y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    # clim = torch.mean(y, dim=(0, 1), keepdim=True)
    clim = clim.to(device=y.device).unsqueeze(0)
    pred = pred - clim
    y = y - clim
    loss_dict = {}

    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_prime = pred[:, i] - torch.mean(pred[:, i])
            y_prime = y[:, i] - torch.mean(y[:, i])
            loss_dict[f"acc_{var}_{log_postfix}"] = torch.sum(w_lat * pred_prime * y_prime) / torch.sqrt(
                torch.sum(w_lat * pred_prime**2) * torch.sum(w_lat * y_prime**2)
            )

    loss_dict["acc"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            y_ = y[:, i]  # B, H, W
            error = (torch.mean(pred_, dim=0) - torch.mean(y_, dim=0)) ** 2  # H, W
            error = torch.mean(error * w_lat)
            loss_dict[f"w_nrmses_{var}"] = torch.sqrt(error) / y_normalization
    
    return loss_dict


def lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)
    y_normalization = clim

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=y.dtype, device=y.device)  # (1, H, 1)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            pred_ = pred[:, i]  # B, H, W
            pred_ = torch.mean(pred_ * w_lat, dim=(-2, -1))  # B
            y_ = y[:, i]  # B, H, W
            y_ = torch.mean(y_ * w_lat, dim=(-2, -1))  # B
            error = torch.mean((pred_ - y_) ** 2)
            loss_dict[f"w_nrmseg_{var}"] = torch.sqrt(error) / y_normalization

    return loss_dict


def lat_weighted_nrmse(pred, y, transform, vars, lat, clim, log_postfix):
    """
    y: [B, V, H, W]
    pred: [B V, H, W]
    vars: list of variable names
    lat: H
    """

    nrmses = lat_weighted_nrmses(pred, y, transform, vars, lat, clim, log_postfix)
    nrmseg = lat_weighted_nrmseg(pred, y, transform, vars, lat, clim, log_postfix)
    loss_dict = {}
    for var in vars:
        loss_dict[f"w_nrmses_{var}"] = nrmses[f"w_nrmses_{var}"]
        loss_dict[f"w_nrmseg_{var}"] = nrmseg[f"w_nrmseg_{var}"]
        loss_dict[f"w_nrmse_{var}"] = nrmses[f"w_nrmses_{var}"] + 5 * nrmseg[f"w_nrmseg_{var}"]
    return loss_dict


def remove_nans(pred: torch.Tensor, gt: torch.Tensor):
    # pred and gt are two flattened arrays
    pred_nan_ids = torch.isnan(pred) | torch.isinf(pred)
    pred = pred[~pred_nan_ids]
    gt = gt[~pred_nan_ids]

    gt_nan_ids = torch.isnan(gt) | torch.isinf(gt)
    pred = pred[~gt_nan_ids]
    gt = gt[~gt_nan_ids]

    return pred, gt


def pearson(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"pearsonr_{var}_day_{day}"] = stats.pearsonr(pred_.cpu().numpy(), y_.cpu().numpy())[0]

    loss_dict["pearsonr"] = np.mean([loss_dict[k] for k in loss_dict.keys()])

    return loss_dict


def lat_weighted_mean_bias(pred, y, transform, vars, lat, log_steps, log_days, clim):
    """
    y: [N, T, 3, H, W]
    pred: [N, T, 3, H, W]
    vars: list of variable names
    lat: H
    """

    pred = transform(pred)
    y = transform(y)

    # lattitude weights
    w_lat = np.cos(np.deg2rad(lat))
    w_lat = w_lat / w_lat.mean()  # (H, )
    w_lat = torch.from_numpy(w_lat).unsqueeze(0).unsqueeze(-1).to(dtype=pred.dtype, device=pred.device)  # [1, H, 1]

    loss_dict = {}
    with torch.no_grad():
        for i, var in enumerate(vars):
            for day, step in zip(log_days, log_steps):
                pred_, y_ = pred[:, step - 1, i].flatten(), y[:, step - 1, i].flatten()
                pred_, y_ = remove_nans(pred_, y_)
                loss_dict[f"mean_bias_{var}_day_{day}"] = pred_.mean() - y_.mean()

                # pred_mean = torch.mean(w_lat * pred[:, step - 1, i])
                # y_mean = torch.mean(w_lat * y[:, step - 1, i])
                # loss_dict[f"mean_bias_{var}_day_{day}"] = y_mean - pred_mean

    loss_dict["mean_bias"] = np.mean([loss_dict[k].cpu() for k in loss_dict.keys()])

    return loss_dict
    

def apply_gaussian_filter(inp, sigma=1.0):
    """Applies a 1D Gaussian filter to a tensor.

    Args:
        tensor: Input tensor of shape (32, 1, 1024).
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.

    Returns:
        Filtered tensor of the same shape as the input.
    """
    from scipy.ndimage import gaussian_filter1d
    device = inp.device
    filtered_tensor = gaussian_filter1d(inp.detach().cpu().numpy(), sigma=sigma, axis=-1)
    filtered_tensor = torch.from_numpy(filtered_tensor).to(device)

    return filtered_tensor

def find_non_spatial_patches(out_transformers, patch_idxs, patched_region_size, N_neg):
    """
    Finds non-spatial patches for each anchor patch in a batch.

    Args:
        out_transformers: The output transformers tensor.
        patch_idxs: A list of anchor patch indices for each sample in the batch.
        patched_region_size: The size of the patched region.
        N_neg: The number of negative patches to sample for each sample.

        # 0, 1, 2, 3, 4, 5, 6
        # 7 ,8 ,9 ,10,11,12,13
        # 14,15,16,17,18,19,20
        # 21,22,23,24,25,26,27
    TODO: currently the distance is 1, but we might modify this when changing the resolution of the image from 5.625 to something else

    Returns:
        A tensor of shape (batch_size, N_neg, out_transformers.shape[2]) containing the negative patches.
    """

    batch_size = len(patch_idxs)
    num_patches = out_transformers.shape[1]

    negative_patches = []
    for batch_idx in range(batch_size):
        patch_idx = patch_idxs[batch_idx]
        non_spatial_idxs = []
        for i in range(num_patches):
            if i != patch_idx and not (
                abs(i // patched_region_size[1] - patch_idx // patched_region_size[1]) <= 1
                and abs(i % patched_region_size[1] - patch_idx % patched_region_size[1]) <= 1
            ):
                non_spatial_idxs.append(i)

        neg_patch_idxs = torch.randint(0, len(non_spatial_idxs), (N_neg,))
        negative_patches.append(out_transformers[batch_idx, [non_spatial_idxs[idx] for idx in neg_patch_idxs], :])

    return torch.stack(negative_patches, dim=0)

def compute_contrastive_loss(x, out_transformers, patch_size, y = None, vars = None, num_anchor_patches = 5, N_neg = 23, scale_temp = 0.05):
    """
    Compute the contrastive loss within the transformer output embedding space
    
    x: input tensor
    out_transformers: transformer encoder outputs (embeddings for each patch)
    patch_size: patch size
    y: output tensor
    vars: list of variable names

    num_anchor_patches: number of anchor patches to sample : default 5
    N_neg: number of negative patches to sample for each anchor patch : default 23 -> 32 total patches - 9 adjacent patches to the anchor patch
    scale_temp: temperature scaling factor for contrastive loss: default 0.05

    Returns:
    contrastive_loss: contrastive loss
    """
    
    if y is None and vars is None:
        # we want to apply some contrastive learning as an auxiliary loss by blurring the anchor patches
        # we use the embedding output of the transformer and construct positive and negative pairs
        # positive : gaussian blur of the same patch or spatially close patches
        # negative : different spatial location but far off. (patches are ordered in a 1D array, but we can assume that the original size of the image was 8,14 and patch size 2)
        patched_region_size = (x.shape[2] // patch_size, x.shape[3] // patch_size) # 4, 7
        # first choose a random patch across the batch. we choose one 1 patch at random per sample in the batch
        num_patches = out_transformers.shape[1]
        # patch_idx = torch.randint(0, num_patches, (x.shape[0],)).tolist() # B
        patch_idx = torch.randint(0, num_patches, (x.shape[0], num_anchor_patches)) # B, num_anchor_patches
        con_loss = 0
        for num_anchor in range(num_anchor_patches):
            # anchor_patch = out_transformers[torch.arange(x.shape[0]), patch_idx, :].unsqueeze(1) # B, 1, D
            anchor_patch = out_transformers[torch.arange(x.shape[0]), patch_idx[:, num_anchor], :].unsqueeze(1) # B, 1, D
            
            # the positive pair is the same patch but with some gaussian noise
            # apply 1d gaussian filter to the patch
            positive_patch = apply_gaussian_filter(anchor_patch)
            # negative_patch = self.find_non_spatial_patches(out_transformers, patch_idx, patched_region_size, N_neg) # B, N_neg, D where N_neg corresponds to the number of negative patches to sample for each sample in the batch
            negative_patch = find_non_spatial_patches(out_transformers, patch_idx[:, num_anchor], patched_region_size, N_neg) # B, N_neg, D where N_neg corresponds to the number of negative patches to sample for each sample in the batch
            # we now have the anchor, positive and negative patches. We can now compute the contrastive loss

        # con_loss /= num_anchor_patches # we average the loss over the number of anchor patches

    else:
        # we want to apply contrastive learning as an auxiliary loss by computing positive patches without blurring
        # we try to make use of the frequency and choose the highest freq as the anchor patch, and similar ones as positive patches
        # first we choose the anchor patch. 
        assert num_anchor_patches == 1, "We only support 1 anchor patch for now"
        # assert N_neg == 1, "We only support 1 negative patch for now"

        patch_freq_array = []
        # we only do this for pm2p5
        for idx, var in enumerate(vars):
            if var == 'pm2p5':
                y = y[:, idx].detach().cpu() # B, H, W
                break

        for i in range(0, y.shape[1], patch_size):
            for j in range(0, y.shape[2], patch_size):
                # compute frequency of each pixel in patch_size x patch_size
                bins_ind = np.digitize(y[:, i:i+patch_size, j:j+patch_size], BINS['pm2p5'], right=True) - 1
                bins_ind[bins_ind == len(BINS['pm2p5']) - 1] = len(BINS['pm2p5']) - 2 # incase anything is more than the last bin
                tmp_b = []
                for b in range(y.shape[0]):
                    bin_sum = sum([COUNTS['pm2p5'][i] for i in bins_ind[b].flatten()])
                    tmp_b.append(bin_sum) # 32 -> batch

                patch_freq_array.append(tmp_b) # 28, 32
        
        patch_freq_array = np.array(patch_freq_array)
        patch_freq_array = np.einsum('db->bd', patch_freq_array) # batch first
        


        # we choose num_anchor_patches patches with the highest frequency
        patch_idx = np.argsort(patch_freq_array, -1) # -1 is the axis
        anchor_patch = out_transformers[torch.arange(x.shape[0]), patch_idx[:, 0], :].unsqueeze(1)

        # positive patch is the closest patch to the anchor patch i.e the next highest frequency patch
        positive_patch = out_transformers[torch.arange(x.shape[0]), patch_idx[:, 1], :].unsqueeze(1)

        #negative patch is the lowest frequency patch
        # negative_patch = out_transformers[torch.arange(x.shape[0]), patch_idx[:, :N_neg], :].unsqueeze(1)
        negative_patch = []
        for i in range(x.shape[0]):
            negative_patch.append(out_transformers[i, patch_idx[i, -N_neg:], :])
        negative_patch = torch.stack(negative_patch, dim=0)


        # convert the main variables to the device
        anchor_patch = anchor_patch.to(device=x.device)
        positive_patch = positive_patch.to(device=x.device)
        negative_patch = negative_patch.to(device=x.device)

    cos_sim = torch.nn.CosineSimilarity(dim=-1)
    pos_sim = cos_sim(anchor_patch, positive_patch)
    N_neg = negative_patch.shape[1] # number of negative pairs
    
    # we compute the negative similarity for all negative pairs and one anchor
    neg_sim = cos_sim(anchor_patch.repeat(1, N_neg, 1), negative_patch)
    neg_sim = neg_sim.reshape(-1, N_neg)

    # compute the loss
    scale_temp = 0.05
    loss = -torch.sum(torch.log(torch.exp(pos_sim/scale_temp) / (torch.exp(pos_sim/scale_temp) + torch.sum(torch.exp(neg_sim/scale_temp), dim=1)))) / anchor_patch.shape[0]

    return loss

