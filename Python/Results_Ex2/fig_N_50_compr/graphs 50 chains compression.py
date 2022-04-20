# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 16:59:09 2021

@author: ECURBELO
"""
import os
from os.path import abspath,split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")


path,file = split(abspath(__file__))
files = os.listdir(path)
files.remove(file)

if "imagenes" not in files:
    os.mkdir("imagenes")

data = pd.read_csv("sim.csv")

data_no_compression = pd.read_csv("sin comprimir.csv")
for i in range(len(data_no_compression["Parameters"])):
    aux = eval(data_no_compression["Parameters"][i])
    aux = (aux[0],aux[1])
    data_no_compression["Parameters"][i] = aux
    
denominators = ["spatial","temporal","all"]

for i in range(len(data["Parameters"])):
    aux = eval(data["Parameters"][i])
    aux = (aux[0],aux[1])
    data["Parameters"][i] = aux
    
    
parameters = [(0.25,1)]

cluster_numbers = [3,21,50,200]
for par in parameters:
    means = []
    labels= []
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # dealing with LAIS
    aux_lais = data[data["Parameters"]==par]
    which_epochs = (aux_lais["Epochs"].value_counts()).index
    which_dens = (aux_lais["Denominator"].value_counts()).index
    for k in cluster_numbers:
    
        aux = aux_lais[aux_lais["Clusters"]==k]
        
        mean_error = aux["Errors"].mean()
        label = f"CLAIS, {k} clusters"
        
        means.append(mean_error)
        labels.append(label)
            
    for den in denominators:
        
        lais_den = data_no_compression[(data_no_compression["Parameters"]==par)&(data_no_compression["Algorithm"]=="LAIS")&(data_no_compression["Denominator"]==den)]
        
        means_ = lais_den["Errors"].mean()
        
        if den != "all":
            
            label = f"LAIS, denominator {den}"
            
        else:
            
            label = "LAIS, total denominator"
        
        means.append(means_)
        
        labels.append(label)
        
    ham_aux = data_no_compression[(data_no_compression["Parameters"]==par)&(data_no_compression["Algorithm"]=="HAM")&(data_no_compression["Epochs"]==48)]
        
    mean_ = ham_aux["Errors"].mean()
    
    means.append(mean_)
    
    label = "Hamiltonian"
    
    labels.append(label)    
        
    #dealing whit HAM
    # aux_ham = data[(data["Algorithm"]=="HAM")&(data["Parameters"]==par)]
    # which_epochs = (list((aux_ham["Epochs"].value_counts()).index))
    # which_epochs.sort()
    # for e in which_epochs:
    #     aux = aux_ham[aux_ham["Epochs"]==e]
        
    #     mean_error = aux["Errors"].mean()
    #     label = f"HAM, {e} épocas"
        
    #     means.append(mean_error)
    #     labels.append(label)
    
    colors = ["tab:blue","tab:orange","tab:cyan","tab:grey","greenyellow","deepskyblue",
              "goldenrod","orangered"]
    
    for i in range(len(means)):
        
        b=ax.bar(i+1,means[i],label=labels[i],width = 0.5)
    
        b[0].set_color(colors[i])
        
    # ax.set_title(f"{par}",fontsize=20)
    
    # ax.legend(loc="lower right",bbox_to_anchor=(1.55,0.0))
    
    ax.legend(loc="upper left")
    
    fig.savefig(f"imagenes/{par}, 50 cadenas.png",bbox_inches="tight")

