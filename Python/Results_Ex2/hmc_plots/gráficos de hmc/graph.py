# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 14:40:37 2021

@author: ECURBELO
"""

import os
from os.path import abspath,split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("default")
# plt.style.use("classic")

path,file = split(abspath(__file__))

files = os.listdir(path)

files.remove(file)

if "imagenes" not in files:
    
    os.mkdir("imagenes")
    
else:
    
    files.remove("imagenes")



def index(obj):
    i = names_as_numbers.index(eval(obj))
    return i

names = [files[i][:-4] for i in range(len(files))]

names_as_numbers = [eval(names[i]) for i in range(len(names))]

names_as_numbers.sort()

names_sorted = sorted(names,key=index)
        
file_names = [i+".csv" for i in names_sorted]

total = 1200

epochs_ham = [(total//N)*2 for N in names_as_numbers]

parameters =  ["(0.25, 1)","(0.5, 1)","(1, 3)","(1, 5)"]
# parameters =  ["(0.25, 1)"]

denominator = ["spatial","temporal","all"]

for par in parameters:
        
    lais_spatial,lais_temporal,lais_all,ham = [],[],[],[]
    
    
    for data_name in file_names:
        
        data = pd.read_csv(data_name)
        
        data_par = data[data["Parameters"]==par]
        
        ind = file_names.index(data_name)
        
        labels = []
        
        # Lais
        for den in denominator:
            
            data_par_lais = data_par[(data_par["Algorithm"]=="LAIS")&(data_par["Denominator"]==den)]

            mean_error = data_par_lais["Errors"].mean()
            
            if den == "spatial":
                
                lais_spatial.append(mean_error)
                
            elif den == "temporal":
                
                lais_temporal.append(mean_error)
                
            else:
                
                lais_all.append(mean_error)
                
            if den != "all":
                labels.append(f"LAIS, {den} denominator")
            else:
                labels.append("LAIS, total denominator")
        
        # HAM
        N = names_as_numbers[ind]
        
        data_par_ham = data_par[(data_par["Algorithm"]=="HAM") & (data_par["Epochs"] == epochs_ham[ind])]
            
        mean_error = data_par_ham["Errors"].mean()
        
        ham.append(mean_error)
        
        labels.append("Hamiltonian")
    
    fig = plt.figure()
    
    ax = fig.add_subplot(111)
    
    ax.plot(names_as_numbers,lais_spatial,"-",label=labels[0],marker="o")
    
    ax.plot(names_as_numbers,lais_temporal,"-.",label=labels[1],marker="*")
    
    ax.plot(names_as_numbers,lais_all,"--",label=labels[2],marker=">")
    
    ax.plot(names_as_numbers,ham,":",label=labels[3],marker="<")

    # plt.legend(bbox_to_anchor=(1.,1.))
    
    plt.legend(loc="best",fontsize=10)
    
    ax.set_xlabel("N",fontsize=15)
    
    ax.set_ylabel("MSE",fontsize=15)
    
    plt.tick_params(labelsize=15)
    # ax.set_title(f"{par}",fontsize=15,pad=15)
    
    fig.savefig(f"imagenes/{par}.png",bbox_inches="tight")
    
    
    
    










