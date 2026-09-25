
from biomed_clip import MediraiBiomedClip
from derm_foundation import MediraiDermFoundation

from .predictor_network import PredictorNet, PredictorNetTrainer

import pandas as pd
import numpy as np

import torch

train_df = pd.read_csv('./train_data_EQ.csv')
found_fusion_feats = []

clip_pred_net = PredictorNet(512, 128) 
mbmc = MediraiBiomedClip(None, 'cpu', clip_pred_net, pred_path = './biomed_clip_predictor_net_epoch-19.plk')

found_pred_net = PredictorNet(6144, 128)
mdf = MediraiDermFoundation(None, 'cpu', found_pred_net, pred_model_path='./derm_foundation_predictor_net_epoch-19.plk')

bmc_feats = np.load('./embs/bmc_embs.npy')
mdf_feats = np.load('./embs/df_embs.npy')

fusion_feats = np.zeros((len(train_df), 128+128))
for i in range(len(train_df)):

    bmc_feat = mbmc.pred_model.get_hidden(torch.from_numpy(bmc_feats[i])).detach().numpy()
    mdf_feat = mdf.pred_model.get_hidden(torch.from_numpy(mdf_feats[i])).detach().numpy()
    
    found_fusion_feat = np.concat((bmc_feat, mdf_feat))
    fusion_feats[i] = found_fusion_feat

np.save('./embs/foundation_fusion_embs.npy', fusion_feats)

print(fusion_feats.shape)

fusion_lbls = np.load('./embs/df_lbls.npy')
fusion_pred_net = PredictorNet(256,128, dropout_prob=0.95)
fusion_trainer = PredictorNetTrainer(fusion_pred_net, fusion_feats, fusion_lbls, 'foundation_fusion', num_epochs=50)
fusion_trainer.train()


#tests

from sklearn.metrics import confusion_matrix

test_df = pd.read_csv('./test_EQ.csv')
fusion_preds = []
clip_preds = []
derm_preds = []
truth = []

fusion_pred_net.eval()

for i in range(len(test_df)):
    
    img_p = test_df.loc[i, 'image_path']
    
    _, clip_emb = mbmc.get_img_features(img_p) 
    derm_emb = mdf.gen_embedding_from_path(img_p)

    bmc_feat = mbmc.pred_model.get_hidden(clip_emb).detach().numpy()
    mdf_feat = mdf.pred_model.get_hidden(torch.from_numpy(derm_emb)).detach().numpy()

    clip_pred = mbmc.predict(img_p).detach().numpy()
    derm_pred = mdf.predict(img_p)

    #print(bmc_feat, mdf_feat)
    fusi_pred = fusion_pred_net.forward(torch.from_numpy(np.concat((bmc_feat, mdf_feat)))).detach().numpy()
    
    fusion_preds.append(np.argmax(fusi_pred))
    clip_preds.append(np.argmax(clip_pred))
    derm_preds.append(np.argmax(derm_pred))
    truth.append(int(test_df.loc[i,'target']))


print('Fusion Results')
print(confusion_matrix(truth, fusion_preds))

print('Clip Results')
print(confusion_matrix(truth, clip_preds))

print('Derm Results')
print(confusion_matrix(truth, derm_preds))



        
