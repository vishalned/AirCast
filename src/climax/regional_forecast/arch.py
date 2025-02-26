# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from climax.arch import ClimaX
import torchvision
import torch.nn.functional as F
from climax.utils.metrics import compute_contrastive_loss
import os

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

BATCH_NUM = 0

class RegionalClimaX(ClimaX):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1, contrastive_loss=False):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, mlp_ratio, drop_path, drop_rate)
        self.contrastive_loss = contrastive_loss

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables, region_info):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # get the patch ids corresponding to the region
        region_patch_ids = region_info['patch_ids']
        x = x[:, :, region_patch_ids, :]

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed[:, region_patch_ids, :]

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x
    



    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, region_info):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            region_info: Containing the region's information

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables, region_info)  # B, L, D

        try: 
            preds_weather = self.head_weather(out_transformers)  # B, L, V*p*p
        except:
            preds_weather = None
        # we now have a new head for chemical variables 
        try:
            preds_chem = self.head_chem(out_transformers)  # B, L, V*p*p
        except:
            preds_chem = None

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']


        if preds_weather is not None:
            preds_weather = self.unpatchify(preds_weather, h = max_h - min_h + 1, w = max_w - min_w + 1)
            out_var_ids_weather = self.get_var_ids(tuple(self.weather_var), preds_weather.device)
            preds_weather = preds_weather[:, out_var_ids_weather]
            y_weather = y[:, out_var_ids_weather, min_h:max_h+1, min_w:max_w+1]

        if preds_chem is not None:
            preds_chem = self.unpatchify(preds_chem, h = max_h - min_h + 1, w = max_w - min_w + 1) 
            out_var_ids_chemical = self.get_var_ids(tuple(self.chemical_var), preds_chem.device) 
            preds_chem = preds_chem[:, out_var_ids_chemical]
            y_chemical = y[:, out_var_ids_chemical, min_h:max_h+1, min_w:max_w+1] 
        
        
        # y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        if metric is None:
            loss = None
        else:
            loss_weather = [m(preds_weather, y_weather, self.weather_var, lat) for m in metric] if preds_weather is not None else None
            loss_chemical = [m(preds_chem, y_chemical, self.chemical_var, lat) for m in metric] if preds_chem is not None else None

            # we merge the two loss dictionaries and for the key "loss" we sum the two losses
            loss = {}
            if loss_weather is not None:
                for key in loss_weather[0].keys():
                    if key != "loss":
                        loss[key] = loss_weather[0][key]
            if loss_chemical is not None:
                for key in loss_chemical[0].keys():
                    if key != "loss":
                        loss[key] = loss_chemical[0][key]

            contrastive_blur = False
            if self.contrastive_loss:
                if contrastive_blur:
                    con_loss = compute_contrastive_loss(x, out_transformers, self.patch_size, num_anchor_patches=1)
                else:
                    con_loss = compute_contrastive_loss(x, out_transformers, self.patch_size, y_chemical, self.chemical_var, num_anchor_patches=1)

            else:
                con_loss = 0
            
            # loss weather and chemical are both mean of loss in each channel (variable) 
            if loss_weather is not None and loss_chemical is not None:
                loss["loss"] = (loss_weather[0]["loss"] + loss_chemical[0]["loss"]) + con_loss
                loss["con_loss"] = con_loss
            elif loss_weather is not None:
                loss["loss"] = loss_weather[0]["loss"] + con_loss
                loss["con_loss"] = con_loss
            elif loss_chemical is not None:
                loss["loss"] = loss_chemical[0]["loss"] + con_loss
                loss["con_loss"] = con_loss

        return loss, preds_weather, preds_chem

    def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix, region_info, visualize_pred=False, lon = None, visualize_period = None):
        _, preds_weather, preds_chem = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat, region_info=region_info) 

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]
        lon = lon[min_w:max_w+1] if lon is not None else None
        clim = clim[:, min_h:max_h+1, min_w:max_w+1]

        # concatenate pred for weather and chemical variables
        preds = torch.cat([preds_weather, preds_chem], dim=1) if preds_weather is not None and preds_chem is not None else preds_weather if preds_weather is not None else preds_chem

        # normal models
        preds = transform(preds) # uncomment this line when you want the normal models
        
        y = transform(y)

        # persistence model
        # for the persistence model, we assume the forecast is the same as the last observation
        # hence preds = x
        # x = x[:, :, min_h:max_h+1, min_w:max_w+1]
        # preds = transform(x)
        # visualize_pred = False # hardcoded for now
        if visualize_pred:
            assert lon is not None
            # we need to keep track of the batch index for visualization. 
            # each batch corresponds to the prediction of each day in a year sequentially.

            global BATCH_NUM
            num_hrs = 8760
            num_shards = 8
            # each batch number cooresponds to one day between 2017-01-01 and 2018-12-31. No leap years.
            # months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
            months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
            days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            day = (BATCH_NUM // 24) + 1 # we add 1 since the prediction is for the next day
            month = 0
            while day > days[month]:
                day -= days[month]
                month += 1
                if month == 12:
                    month = 0
            year = 2017 + BATCH_NUM // (365*24)
            hour = BATCH_NUM % 24
            if day + 1 > days[month]:
                period_text = f'{year}-{months[month+1]}-{1}-{hour}h'
            else:
                period_text = f'{year}-{months[month]}-{day+1}-{hour}h' # prediction text

            # print(period_text, BATCH_NUM)
            
            # visualizing period
            start_period = visualize_period[0].split('-')
            end_period = visualize_period[1].split('-') if len(visualize_period) > 1 else None
            # VISUALIZE_BATCH_NUM = (months.index(start_period[1])) * 24 * (days[months.index(start_period[1]) - 1] if start_period[1] != 'jan' else 0)
            # for each month until the start month, we add the number of hours in that month based on the days in the month
            VISUALIZE_BATCH_NUM = 0
            VISUALIZE_BATCH_NUM_END = 0

            for i in range(months.index(start_period[1])):
                VISUALIZE_BATCH_NUM += 24 * (days[i])
            # we then add the days in the start month
            VISUALIZE_BATCH_NUM += (int(start_period[2])-1) * 24
            # VISUALIZE_BATCH_NUM += (int(start_period[2])-1) * 24
            VISUALIZE_BATCH_NUM += 8760 * (int(start_period[0]) - 2017)

            # VISUALIZE_BATCH_NUM_END = (months.index(end_period[1])) * 24 * (days[months.index(end_period[1]) - 1] if end_period[1] != 'jan' else 0)
            # VISUALIZE_BATCH_NUM_END += (int(end_period[2])-1) * 24
            for i in range(months.index(end_period[1])):
                VISUALIZE_BATCH_NUM_END += 24 * (days[i])
            VISUALIZE_BATCH_NUM_END += (int(end_period[2])-1) * 24
            VISUALIZE_BATCH_NUM_END += 8760 * (int(end_period[0]) - 2017)
            
            # for now, we visualize every 6hours
            if BATCH_NUM >= VISUALIZE_BATCH_NUM and BATCH_NUM <= VISUALIZE_BATCH_NUM_END:
                if BATCH_NUM % 6 == 0:
                    self.visualize_pred_mena(preds, y, out_variables, lat, lon, clim, period_text, path_name='visualizations_test5')
            BATCH_NUM += 1

        return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]
    
    def visualize_pred(self, preds, y, out_variables, lat, lon, clim, log_postfix, path_name='visualizations'):
        # visualize the prediction
        # the input and ouput have been scaled down to a res of 5.625 degree. We want to plot these predictions on a world map. 
        # the preds are only for a region, the rest of the world is filled with zeros.

        os.makedirs(f'./{path_name}', exist_ok=True)
        lat_ = lat # original lat from the dataset
        lon_ = lon # original lon from the dataset

        lat_ = lat_[::-1]*-1
        
        # print('preds shape:', preds.shape)
        preds = preds[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable
        y_ = y[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable


        # first we plot a world map 
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

        # we get the meshgrid with full range of lat and lon for the resolution of 5.625
        lon, lat = np.meshgrid(np.arange(-180, 180, 5.625), np.arange(-90, 90, 5.625))
        x, y = m(lon, lat)



        # create a temp pred of just 0s for the full world map size
        preds_map = np.zeros((32, 64))
        y_map = np.zeros((32, 64))

        # defining the full range of lat and lon for the resolution of 5.625
        lon_range = np.arange(-180, 180, 5.625)
        lat_range = np.arange(-90+5.625/2, 90, 5.625)

        # print(lat_.shape, lon_.shape)
        for ind_i, i in enumerate(lon_):
            for ind_j, j in enumerate(lat_):
                # print(f'lon: {i}, lat: {j}')
                i_lon = i-360 if i > 180 else i
                ind_lon = np.where(lon_range == i_lon)[0][0]
                ind_lat = np.where(lat_range == j)[0][0]
                # print(f'ind_lon: {ind_lon}, ind_lat: {ind_lat}')
                preds_map[ind_lat, ind_lon] = preds[ind_j, ind_i]
                y_map[ind_lat, ind_lon] = y_[ind_j, ind_i]


        ############################################ plot the prediction ############################################
        # m.pcolormesh(x, y, preds_map, cmap='YlOrRd')
        # m.drawcountries()
        # m.drawcoastlines()
        # m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        # m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        # plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # # plt.title('Predictions (MENA region)')
        # plt.savefig(f'./visualizations/{log_postfix}-prediction.png')
        # plt.savefig(f'./visualizations/{log_postfix}-prediction.pdf', dpi=300, bbox_inches='tight')
        # plt.close()
        ############################################ plot the prediction ############################################
        
        ##################################### plot the error ############################################
        # create a custom color map based on fixed error values ranges i.e 0 in the middle and -0.1 to 0.1 on the edges
        bounds = np.array([-4e-8, -3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8, 4e-8])
        norm = plt.Normalize(bounds.min(), bounds.max())
        c_map = plt.cm.RdBu_r
        colors = c_map(norm(bounds))


        # plot the prediction - actual
        plt.rcParams.update({'font.size': 40})
        plt.figure(figsize=(28, 22))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

        m.pcolormesh(x, y, preds_map - y_map, cmap='RdBu_r', norm=norm)
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0]) 
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])

        plt.colorbar(label='Error in PM2.5 Concentration', shrink=0.63, fraction=0.046, pad=0.05)
        
        # plt.savefig(f'./{path_name}/{log_postfix}-error.png', dpi=300, bbox_inches='tight')
        # plt.savefig(f'./{path_name}/{log_postfix}-error.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the error ############################################

        ##################################### plot the actual value ############################################
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
        m.pcolormesh(x, y, y_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Actual (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.png')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the actual value ############################################
    

    def visualize_pred_china(self, preds, y, out_variables, lat, lon, clim, log_postfix, path_name='visualizations'):
        # visualize the prediction
        # the input and ouput have been scaled down to a res of 5.625 degree. We want to plot these predictions on a world map. 
        # the preds are only for a region, the rest of the world is filled with zeros.

        os.makedirs(f'./{path_name}', exist_ok=True)

        lat_ = lat # original lat from the dataset
        lon_ = lon # original lon from the dataset

        lat_ = lat_[::-1]*-1
        
        # print('preds shape:', preds.shape)
        preds = preds[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable
        y_ = y[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable

        # first we plot a world map 
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

        # we get the meshgrid with full range of lat and lon for the resolution of 5.625
        lon, lat = np.meshgrid(np.arange(-180, 180, 5.625), np.arange(-90, 90, 5.625))
        x, y = m(lon, lat)



        # create a temp pred of just 0s for the full world map size
        preds_map = np.zeros((32, 64))
        y_map = np.zeros((32, 64))

        # defining the full range of lat and lon for the resolution of 5.625
        lon_range = np.arange(-180, 180, 5.625)
        lat_range = np.arange(-90+5.625/2, 90, 5.625)

        # print(lat_.shape, lon_.shape)
        for ind_i, i in enumerate(lon_):
            for ind_j, j in enumerate(lat_):
                # print(f'lon: {i}, lat: {j}')
                i_lon = i-360 if i > 180 else i
                ind_lon = np.where(lon_range == i_lon)[0][0]
                ind_lat = np.where(lat_range == j)[0][0]
                # print(f'ind_lon: {ind_lon}, ind_lat: {ind_lat}')
                preds_map[ind_lat, ind_lon] = preds[ind_j, ind_i]
                y_map[ind_lat, ind_lon] = y_[ind_j, ind_i]


        ############################################ plot the prediction ############################################
        m.pcolormesh(x, y, preds_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Predictions (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.png')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ############################################ plot the prediction ############################################
        

        ##################################### plot the error ############################################
        # create a custom color map based on fixed error values ranges i.e 0 in the middle and -0.1 to 0.1 on the edges
        # bounds = np.array([-4e-8, -3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8, 4e-8])
        bounds = np.array([-3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8])
        norm = plt.Normalize(bounds.min(), bounds.max())
        c_map = plt.cm.RdBu_r
        colors = c_map(norm(bounds))


        # plot the prediction - actual
        plt.rcParams.update({'font.size': 40})
        plt.figure(figsize=(28, 22))
        m = Basemap(projection='cyl', llcrnrlat=3, urcrnrlat=68, llcrnrlon=65, urcrnrlon=155)

        m.pcolormesh(x, y, preds_map - y_map, cmap='RdBu_r', norm=norm)
        m.drawcountries()
        m.drawcoastlines()

        #east asia
        m.llcrnrlon = 65
        m.llcrnrlat = 3
        m.urcrnrlon = 155
        m.urcrnrlat = 68

        # east asia
        m.drawparallels(np.arange(0, 71, 10), labels=[1,0,0,0])
        m.drawmeridians(np.arange(65, 156, 20), labels=[0,0,0,1])

        plt.colorbar(label='Error in PM2.5 Concentration', shrink=0.63, fraction=0.046, pad=0.05)
        
        # plt.title('Prediction - Actual (MENA region) ')
        plt.savefig(f'./{path_name}/{log_postfix}-error.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./{path_name}/{log_postfix}-error.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the error ############################################

        ##################################### plot the actual value ############################################
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
        m.pcolormesh(x, y, y_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Actual (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.png')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the actual value ############################################


    def visualize_pred_mena(self, preds, y, out_variables, lat, lon, clim, log_postfix, path_name='visualizations'):
        # visualize the prediction
        # the input and ouput have been scaled down to a res of 5.625 degree. We want to plot these predictions on a world map. 
        # the preds are only for a region, the rest of the world is filled with zeros.

        os.makedirs(f'./{path_name}', exist_ok=True)


        lat_ = lat # original lat from the dataset
        lon_ = lon # original lon from the dataset

        lat_ = lat_[::-1]*-1
        
        # print('preds shape:', preds.shape)
        preds = preds[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable
        y_ = y[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable


        # first we plot a world map 
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

        # we get the meshgrid with full range of lat and lon for the resolution of 5.625
        lon, lat = np.meshgrid(np.arange(-180, 180, 5.625), np.arange(-90, 90, 5.625))
        x, y = m(lon, lat)

        # create a temp pred of just 0s for the full world map size
        preds_map = np.zeros((32, 64))
        y_map = np.zeros((32, 64))

        # defining the full range of lat and lon for the resolution of 5.625
        lon_range = np.arange(-180, 180, 5.625)
        lat_range = np.arange(-90+5.625/2, 90, 5.625)

        # print(lat_.shape, lon_.shape)
        for ind_i, i in enumerate(lon_):
            for ind_j, j in enumerate(lat_):
                # print(f'lon: {i}, lat: {j}')
                i_lon = i-360 if i > 180 else i
                ind_lon = np.where(lon_range == i_lon)[0][0]
                ind_lat = np.where(lat_range == j)[0][0]
                # print(f'ind_lon: {ind_lon}, ind_lat: {ind_lat}')
                preds_map[ind_lat, ind_lon] = preds[ind_j, ind_i]
                y_map[ind_lat, ind_lon] = y_[ind_j, ind_i]


        ############################################ plot the prediction ############################################
        m.pcolormesh(x, y, preds_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Predictions (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.png')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ############################################ plot the prediction ############################################



        ##################################### plot the error ############################################
        # create a custom color map based on fixed error values ranges i.e 0 in the middle and -0.1 to 0.1 on the edges
        bounds = np.array([-4e-8, -3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8, 4e-8])
        # bounds = np.array([-3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8])
        norm = plt.Normalize(bounds.min(), bounds.max())
        c_map = plt.cm.RdBu_r
        colors = c_map(norm(bounds))
        # plot the prediction - actual
        plt.rcParams.update({'font.size': 40})
        plt.figure(figsize=(28, 22))
        m = Basemap(projection='cyl', llcrnrlat=-5, urcrnrlat=48, llcrnrlon=-10, urcrnrlon=86) # mena

        m.pcolormesh(x, y, preds_map - y_map, cmap='RdBu_r', norm=norm)
        m.drawcountries()
        m.drawcoastlines()
        
        # mena
        m.llcrnrlon = -10
        m.llcrnrlat = -5
        m.urcrnrlon = 86
        m.urcrnrlat = 48
        # mena
        m.drawparallels(np.arange(0, 51, 10), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-10, 91, 20), labels=[0,0,0,1])


        plt.colorbar(label='Error in PM2.5 Concentration', shrink=0.63, fraction=0.046, pad=0.05)
        plt.savefig(f'./{path_name}/{log_postfix}-{BATCH_NUM}-error.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./{path_name}/{log_postfix}-{BATCH_NUM}-error.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the error ############################################

        ##################################### plot the actual value ############################################
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
        m.pcolormesh(x, y, y_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Actual (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.png')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the actual value ############################################



    
    def visualize_pred_usa(self, preds, y, out_variables, lat, lon, clim, log_postfix, path_name='visualizations'):
        # visualize the prediction for north america
        # the input and ouput have been scaled down to a res of 5.625 degree. We want to plot these predictions on a world map. 
        # the preds are only for a region, the rest of the world is filled with zeros.

        os.makedirs(f'./{path_name}', exist_ok=True)


        lat_ = lat # original lat from the dataset
        lon_ = lon # original lon from the dataset

        lat_ = lat_[::-1]*-1
        
        # print('preds shape:', preds.shape)
        preds = preds[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable
        y_ = y[0, -5, :, :].detach().cpu().numpy() # for testing we only take the first batch, and pm2.5 variable


        # first we plot a world map 
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)

        # we get the meshgrid with full range of lat and lon for the resolution of 5.625
        lon, lat = np.meshgrid(np.arange(-180, 180, 5.625), np.arange(-90, 90, 5.625))
        x, y = m(lon, lat)

        # create a temp pred of just 0s for the full world map size
        preds_map = np.zeros((32, 64))
        y_map = np.zeros((32, 64))

        # defining the full range of lat and lon for the resolution of 5.625
        lon_range = np.arange(-180, 180, 5.625)
        lat_range = np.arange(-90+5.625/2, 90, 5.625)

        # print(lat_.shape, lon_.shape)
        for ind_i, i in enumerate(lon_):
            for ind_j, j in enumerate(lat_):
                # print(f'lon: {i}, lat: {j}')
                i_lon = i-360 if i > 180 else i
                ind_lon = np.where(lon_range == i_lon)[0][0]
                ind_lat = np.where(lat_range == j)[0][0]
                # print(f'ind_lon: {ind_lon}, ind_lat: {ind_lat}')
                preds_map[ind_lat, ind_lon] = preds[ind_j, ind_i]
                y_map[ind_lat, ind_lon] = y_[ind_j, ind_i]


        ############################################ plot the prediction ############################################
        m.pcolormesh(x, y, preds_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Predictions (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.png')
        plt.savefig(f'./{path_name}/{log_postfix}-prediction.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ############################################ plot the prediction ############################################



        ##################################### plot the error ############################################
        # create a custom color map based on fixed error values ranges i.e 0 in the middle and -0.1 to 0.1 on the edges
        # bounds = np.array([-4e-8, -3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8, 4e-8])
        bounds = np.array([-3e-8, -2e-8, -1e-8, 0, 1e-8, 2e-8, 3e-8])
        norm = plt.Normalize(bounds.min(), bounds.max())
        c_map = plt.cm.RdBu_r
        colors = c_map(norm(bounds))
        # plot the prediction - actual
        plt.rcParams.update({'font.size': 40})
        plt.figure(figsize=(28, 22))
        m = Basemap(projection='cyl', llcrnrlat=10, urcrnrlat=70, llcrnrlon=-140, urcrnrlon=-60)

        m.pcolormesh(x, y, preds_map - y_map, cmap='RdBu_r', norm=norm)
        m.drawcountries()
        m.drawcoastlines()
        # usa
        m.llcrnrlon = 215
        m.llcrnrlat = 10
        m.urcrnrlon = 305
        m.urcrnrlat = 70

        # usa
        m.drawparallels(np.arange(0, 71, 10), labels=[1,0,0,0])
        m.drawmeridians(np.arange(215, 306, 20), labels=[0,0,0,1])



        plt.colorbar(label='Error in PM2.5 Concentration', shrink=0.63, fraction=0.046, pad=0.05)
        plt.savefig(f'./{path_name}/{log_postfix}-error.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'./{path_name}/{log_postfix}-error.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        plt.clf()
        ##################################### plot the error ############################################

        ##################################### plot the actual value ############################################
        plt.rcParams.update({'font.size': 10})
        plt.figure(figsize=(10, 8))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180)
        m.pcolormesh(x, y, y_map, cmap='YlOrRd')
        m.drawcountries()
        m.drawcoastlines()
        m.drawparallels(np.arange(-90, 91, 30), labels=[1,0,0,0])
        m.drawmeridians(np.arange(-180, 181, 60), labels=[0,0,0,1])

        # reset the font size and figure size
       
        plt.colorbar(label='PM2.5 Concentration (kg/m3)')
        # plt.title('Actual (MENA region)')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.png')
        plt.savefig(f'./{path_name}/{log_postfix}-actual.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        ##################################### plot the actual value ############################################





