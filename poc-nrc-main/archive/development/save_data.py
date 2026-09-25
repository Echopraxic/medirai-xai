import shutil
import pandas as pd


train = pd.read_csv('./train_data_TESTING_EQ.csv')
test = pd.read_csv('./test_EQ.csv')

for i, row in train.iterrows():

    f_path = row['image_path']

    if 'hand' in f_path:
        f_path = f_path.replace('../', '../../../data/')
    else:
        f_path = f_path.replace('../data/isic_archive', '../../../data')
    
    print(f_path)
    target = int(row['target'])
    dest = f'./train/{target}/{f_path.split("/")[-1]}'
    shutil.copyfile(f_path, dest)

for i, row in test.iterrows():

    f_path = row['image_path']
    target = row['target']
    dest = f'./test/{int(target)}/{f_path.split("/")[-1]}'
    print(f_path, '->', dest)
    shutil.copyfile(f_path, dest)
