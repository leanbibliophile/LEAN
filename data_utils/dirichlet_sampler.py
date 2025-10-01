#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np
from torchvision import datasets, transforms
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from collections import Counter
from copy import deepcopy

def dirichlet_sampler(dataset, num_users, prior_distribution=None, diric=100, items_per_user=500, plot=False):
    # TODO Make items_per_user also inputable as an array for non-identical quantity as well
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    
    if prior_distribution is None:
        prior_distribution = [1 for i in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #items_per_user = dataset_size / num_users # Make sure this is an integer?

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if plot: 
        hsv = plt.get_cmap('hsv')
        color_num = num_classes
        plot_colors = hsv(np.linspace(0, 1.0, color_num))
        space = [0.0 for i in range(num_users)]
        for i in range(num_classes):
            plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
            space += distributions[i]
        plt.savefig(f'./diric_users{num_users}_dir{diric}_distribution2.png')

    distributions = distributions.transpose()

    for i in range(num_users):
        images_distribution = np.round((items_per_user * distributions[i])).astype(int)
        images_distribution[-1] = max(items_per_user - sum(images_distribution[0:-1]), 0) # Maybe we'll get extras but it is fine
        for j in range(len(images_distribution)):
            idxs_set = idxs_labels[0, np.where(idxs_labels[1, :] == j)][0].tolist()
            #print(f"User {i} has chosen {images_distribution[j]} samples for class {j}")
            dict_users[i] = np.concatenate((dict_users[i], np.random.choice(idxs_set, images_distribution[j], replace=True)), axis=0)
    
    return dict_users

def dirichlet_sampler_noreplace(dataset, num_users, prior_distribution=None, diric=100, plot=False):
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)
    init_sample_size = dataset_size // num_users
    remain = dataset_size % num_users
    items_per_user = [init_sample_size] * num_users
    for i in range(remain):
        items_per_user[i] += 1

    if prior_distribution is None:
        prior_distribution = [1 for i in range(num_classes)]
    distributions = np.random.dirichlet(diric * np.array(prior_distribution), num_users).transpose()

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    #items_per_user = dataset_size / num_users # Make sure this is an integer?

    idxs = np.arange(dataset_size)
    labels = np.array(dataset.targets)
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    if plot: 
        hsv = plt.get_cmap('hsv')
        color_num = num_classes
        plot_colors = hsv(np.linspace(0, 1.0, color_num))
        space = [0.0 for i in range(num_users)]
        for i in range(num_classes):
            plt.barh(range(num_users), distributions[i], left=space, color=plot_colors[i])
            space += distributions[i]
        plt.savefig(f'./diric_users{num_users}_dir{diric}_distribution2.png')

    distributions = distributions.transpose()

    index_mapping = {}
    avail_class = set(range(num_classes))
    for c in range(num_classes):
        index_mapping[c] = set(idxs_labels[0, np.where(idxs_labels[1, :] == c)][0].tolist())

    order = np.random.permutation(num_users) # Avoid biasing to early id clients, not that this should matter
    for idx, id in enumerate(order):
        initial_distr = distributions[id]
        items = items_per_user[id]

        chosen = 0
        
        if idx == len(order) - 1:
            for c in avail_class:
                add_items = list(index_mapping[c])
                dict_users[id] = np.append(dict_users[id], add_items)
        else:
            initial_distr = distributions[id]
            while chosen < items:
                avail_choices = list(avail_class)
                updated_distr = initial_distr[avail_choices]
                updated_norm = sum(updated_distr)
                updated_distr = updated_distr / updated_norm

                c = np.random.choice(avail_choices, p=updated_distr)   
                selection = list(index_mapping[c])
                choice = np.random.choice(selection)  

                index_mapping[c].remove(choice)
                dict_users[id] = np.append(dict_users[id], [choice])

                if len(index_mapping[c]) == 0:
                    avail_class.remove(c)
                chosen += 1
                
    return dict_users

def extreme_sampling(dataset, num_users):
    dataset_size = len(dataset)
    num_classes = len(dataset.classes)