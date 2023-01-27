#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 18:55:43 2023

@author: balthazar
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import warnings
from console_progressbar import ProgressBar

def get_neighbor_seeds(df, seed_list):
    nbr_indices = []
    nbr_values = []
    for seed in seed_list:
        xi = df.loc[seed]['rp']
        yi = df.loc[seed]['dp']
        nbr_indices.append(df[(df['rp']==xi+1)*(df['dp']==yi)].index.values.tolist())
        nbr_indices.append(df[(df['rp']==xi)*(df['dp']==yi+1)].index.values.tolist())
        nbr_indices.append(df[(df['rp']==xi-1)*(df['dp']==yi)].index.values.tolist())
        nbr_indices.append(df[(df['rp']==xi)*(df['dp']==yi-1)].index.values.tolist())
    
    nbr_indices = [item for sublist in nbr_indices for item in sublist]
    nbr_values.append([df.loc[idx]['vp'] for idx in nbr_indices])
    return np.array(nbr_indices), np.array(nbr_values)[0]
  
def fof(df,verbose=False,plot=False,snr=3.0,hdu=None,min_size=4,dist_mult=1,iters=0):
    groups = []
    groups_flat = []
    df = df[df['snr']>snr].sort_values('tp',ascending=False)
    gdf = df.copy()
    gdf[['gn']] = 0
    pb = ProgressBar(total=df['tp'].max(),prefix='Finding groups...',suffix='Completed',decimals=0,length=50,fill='>',zfill=' ')
    gn = 0
    for i in df.index.values:
        pb.print_progress_bar(df['tp'].max()-df.loc[i]['tp'])
        if i in groups_flat:
            continue
        flag = 0
        seeds = [i]
        group = [i]
        
        while flag==0:
            idx, vals = get_neighbor_seeds(df, seeds)
            vdist = np.abs(df.loc[i]['vp']-vals)
            #k = idx[vdist<dist_mult*df.loc[i]['sd']].tolist()
            #unique_k = list(set(k)) # Remove duplicate neighbors from different seeds
            #new_idx = [idx for idx in unique_k if idx not in group+groups_flat] # Remove seeds existing in group or groups

            tdf = df.loc[idx].copy()
            tdf['vdist'] = vdist
            tdf = tdf[tdf['vdist']<dist_mult*df.loc[i]['sd']]
            tdf = tdf.drop_duplicates().sort_values('vdist') # Remove duplicate neighbors from different seeds
            tdf = tdf.drop_duplicates(['rp','dp'],keep='first') # Keep closest seed from same neighbor
            new_idx = [idx for idx in tdf.index.values if idx not in group+groups_flat] # Remove seeds existing in group or groups
            #print(set(new_idx1)==set(new_idx))
            if plot==True:
                mask = np.zeros(hdu.data.shape)
                coords = np.vstack([df.loc[group]['rp'].values,df.loc[group]['dp'].values])
                mask[coords[1,:], coords[0,:]] = 1.0
                plt.subplot(1,1,1,projection=WCS(hdu.header))
                plt.imshow(hdu.data)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="No contour levels were found within the data range.")
                    plt.contour(mask,levels=[1],colors='red',linewidths=3)
                plt.pause(0.001)
                plt.clf()

            if verbose==True:
                #print('dist',vdist,' \ngroup',group,'\nnew_idx',new_idx,'\nk\n\n',k)
                print('\nnew_idx',new_idx,'\ngroup',group,'\ngroups',groups,'\ngroup+groups',group+groups_flat)

            if new_idx == [] and get_group_size(df,group)[0]>min_size-1:
                gn+=1
                groups.append(group)
                groups_flat = groups_flat+group
                gdf.loc[group,'gn']=gn
                flag=1
            elif new_idx != []:
                seeds = new_idx
                group += new_idx
            else:
                break

    groups.sort(key=len,reverse=True)
    print('\n\n############################################')
    print(f'\nGroups found: {len(groups)}')
    print('\n############################################')
    
    return groups, gdf

def get_group_size(df, group=None):
    if type(group) == int:
        unique_coords = df[df['gn']==group][['rp','dp']].drop_duplicates().values
    elif type(group) == list:
        unique_coords = df.loc[group][['rp','dp']].drop_duplicates().values
    else:
        raise Exception('group should either be int or list!')
    unique_len = len(unique_coords)
    
    return unique_len, unique_coords

def get_group_mask(df,group,hdu):
    mask = np.zeros(hdu.data.shape)
    coords = get_group_size(df,group)[1]
    mask[coords[:,1],coords[:,0]]=1
    
    return mask