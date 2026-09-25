import pandas as pd
import numpy as np


def make_mixup_data_train_csv(standard_train_csv : str, 
                              num_mixups_per_class : int = 50_000, 
                              outname : str ='./mixup_train_data.csv') -> str:
    '''
    Helper function to create the data mixup training to_csv. A mixup
    example is create by taking two observations in the same class
    (obs1, obs2) and creating a linear combination, using lambda randomly
    selected between 0.3 and 0.6. Thus, the new obs is:
        obs_new = lambda*obs1 + (1-lambda)*obs2
    where addition is completed pixelwise.

    Parameters
    ----------
    standard_train_csv : str
        path to a 'standard' training csv. This would be one that contains
        columns 'image_id', 'image_path', and 'target'
    num_mixups_per_class : int
        How many examples using the mixup to create for each class. A total
        of 2*num_mixups_per_class observation will be created. Do not exceed
        (total example of smallest class) choose (2).
    outname : str
        Location to save the training csv. The csv should always be saved 
        for reproducibility

    Returns
    -------
    outname : str
        path to saved data mixup training csv.
    '''
    
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
        
        all_mixups = []
        for i in range(mat.shape[0]):
            for j in range(i, mat.shape[1]):
                all_mixups.append( (i,j,mat[i,j]) )

        mixups_idx = np.random.choice(np.arange(0,len(all_mixups)), size=num_obs, replace=False)
        #np.random.choice(all_mixups, size=num_obs, replace=False)
        
        data = []
        for k in mixups_idx:
            i, j, lam = all_mixups[k]
            data.append({
                 'image_id_1':df.loc[i, 'image_id'],
                 'image_id_2':df.loc[j, 'image_id'],
                 'image_path_1': df.loc[i, 'image_path'],
                 'image_path_2': df.loc[j, 'image_path'],
                 'lambda':lam,
                 'target':target
            })
            
        return pd.DataFrame(data)

    print('Creating positive mixup matrix...')
    pos_mat = gen_mixup_mat(len(pos_df))
    print('Creating negative mixup matrix...')
    neg_mat = gen_mixup_mat(len(neg_df))

    print('Getting positive mixup examples...')
    pos_mixup_df = flatten_mixup_mat(pos_mat, pos_df, 1)
    print('Getting negative mixup examples...')
    neg_mixup_df = flatten_mixup_mat(neg_mat, neg_df, 0)

    print('Making single training csv...')
    mixup_df = pd.concat((pos_mixup_df, neg_mixup_df), ignore_index=True)
    mixup_df = mixup_df.sample(frac=1.0)
    mixup_df.to_csv(outname, index=False)
    print('\t...done!')

    return outname

if __name__ == '__main__':

    base_train_csv = './train_test_csvs/train_data_EQ_new_negs.csv'
    make_mixup_data_train_csv(base_train_csv, 1_500_000, './mixup_train_data_3M.csv')


