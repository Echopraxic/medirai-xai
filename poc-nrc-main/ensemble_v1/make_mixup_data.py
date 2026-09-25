import pandas as pd
import numpy as np


def make_mixup_data_train_csv(standard_train_csv, num_mixups_per_class=50_000, outname='./mixup_train_data.csv'):
    
    print('Creating mixup training csv...')
    data_df = pd.read_csv(standard_train_csv)
    pos_df = data_df.loc[data_df['target'] == 1].reset_index()
    neg_df = data_df.loc[data_df['target'] == 0].reset_index()

    assert len(pos_df) == len(neg_df)

    def gen_mixup_mat(num):
          
        mixup_mat = np.zeros((num, num))

        for i in range(mixup_mat.shape[0]):
            for j in range(i, mixup_mat.shape[1]):
                    mixup_mat[i,j] = np.random.uniform(low=0.3, high=0.7)
                    if i == j:
                        mixup_mat[i,j] = 1

        return mixup_mat


    def flatten_mixup_mat(mat, df, target, num_obs=num_mixups_per_class):
         
        selections = []
        data = []
        while len(selections) < num_obs:

            i = np.random.randint(low=0, high=len(df))
            j = np.random.randint(low=i, high=len(df))

            while (i,j) in selections:
                i = np.random.randint(low=0, high=len(df))
                j = np.random.randint(low=i, high=len(df))

            selections.append((i,j))
            data.append({
                 'image_id_1':df.loc[i, 'image_id'],
                 'image_id_2':df.loc[j, 'image_id'],
                 'image_path_1': df.loc[i, 'image_path'],
                 'image_path_2': df.loc[j, 'image_path'],
                 'lambda':mat[i,j],
                 'target':target
            })
            
        return pd.DataFrame(data)

    pos_mat = gen_mixup_mat(len(pos_df))
    neg_mat = gen_mixup_mat(len(neg_df))

    pos_mixup_df = flatten_mixup_mat(pos_mat, pos_df, 1)
    neg_mixup_df = flatten_mixup_mat(neg_mat, neg_df, 0)

    mixup_df = pd.concat((pos_mixup_df, neg_mixup_df), ignore_index=True)
    mixup_df = mixup_df.sample(frac=1.0)
    mixup_df.to_csv(outname, index=False)
    print('\t...done!')

    return outname

