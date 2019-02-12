import pandas as pd
import networkx as nx
import os

def process_cooja2_traces(path, tracemask):
    for file in os.listdir(path):
        if file.startswith(tracemask):
            print(file)




process_cooja2_traces("cooja-9nodes-0.2/traces/normal",'grid9_normal_2019-02-11_20:22:01_')