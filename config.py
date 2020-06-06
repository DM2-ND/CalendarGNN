"""
Configurations
"""

import os

''' Dirs '''
dataset_dir = os.path.join('data', 'w1')
# dataset_dir = os.path.join('data', 'w2')

''' Files '''
users_info_file = os.path.join(dataset_dir, 'users_info.csv')
sessions_info_file = os.path.join(dataset_dir, 'sessions_info.csv')
locations_info_file = os.path.join(dataset_dir, 'locations_info.csv')
items_info_file = os.path.join(dataset_dir, 'items_info.csv')

user2sessions_file = os.path.join(dataset_dir, 'user2sessions.csv')
session2items_file = os.path.join(dataset_dir, 'session2items.csv')


best_model_file = os.path.join('.', 'best_model.pt')

''' Parameters '''
rand_seed = 999
