
from medirai_base_model import MediraiEnsembleModelV1

class MediraiDNNExplainers:
    '''
    Helper class for producing explainability figures.

    Attributes
    ----------
    dnns : MediraiEnsembleModelV1
        instance of dnns to be explained.

    Important
    ---------
    This class can be modified to work as RestAPI. Figures would
    need to be written to buffers (encoded as pngs) to be ready for
    transmission
    '''
    def __init__(self, dnn_model_paths, device='cpu'):
        #device='cpu', use_hidden_layers=True 
        #self.dnn_model_paths = dnn_model_paths
        self.dnns : MediraiEnsembleModelV1 = MediraiEnsembleModelV1(device=device)
        self._load_models(dnn_model_paths)
       

    def _load_models(self, dnn_model_paths : list[str]) -> None:
        '''
        Helper function to load all the models from frozen state dictionaries. 

        Parameters
        ----------
        dnn_model_paths : list[str]
             list of frozen state dictionaries to be loaded.
        ''' 
        self.dnns.load(path_to_densenet=dnn_model_paths[0])
        self.dnns.load(path_to_efficientnet=dnn_model_paths[1])
        self.dnns.load(path_to_inception=dnn_model_paths[2])
        self.dnns.load(path_to_resnet=dnn_model_paths[3])

    def get_all_explainers(self, img : str, target_class : int, base_outname: str = './explainer.png'):
        '''
        Produce all three explainabilities for all four models provided a single 
        image path.

        Parameters
        ----------
        img : str
            path to image to be explained
        target_class : int
            for gradcam explainer to work properly a class needs to be provided. Must
            be 0 or 1. 
        base_outname : str
            sample output name and path to be modified to save all results. Must end
            in ".png"
        '''
        
        if base_outname[-4:] != '.png':
            raise ValueError(f'Keyword argument "base_outname" must end in ".png", but ends in {base_outname[-4:]}')

        self.get_gradcam_explainers(img, target_class, base_outname=base_outname.replace('.png', '_gradcam.png'))
        self.get_shap_explainers(img, base_outname=base_outname.replace('.png', '_shap.png'))
        self.get_lime_explainers(img, base_outname=base_outname.replace('.png', '_lime.png'))
       
    def get_gradcam_explainers(self, img : str, target_class : int, base_outname='./explainer_gradcam.png'):
        '''
        Generate gradcam explainers for all four model for the provided input image.

        Parameters
        ----------
        img : str
            path to image to be explained
        target_class : int
            for gradcam explainer to work properly a class needs to be provided. Must
            be 0 or 1. 
        base_outname : str
            sample output name and path to be modified to save all results. Must end
            in ".png"
        '''
        if base_outname[-4:] != '.png':
            raise ValueError(f'Keyword argument "base_outname" must end in ".png"')

        self.dnns.densenet.predict_with_cam(img, target_class, base_outname.replace('.png', '_densenet.png'))
        self.dnns.efficientnet.predict_with_cam(img, target_class, base_outname.replace('.png', '_efficientnet.png'))
        self.dnns.inception.predict_with_cam(img, target_class, base_outname.replace('.png', '_inception.png'))
        self.dnns.resnet.predict_with_cam(img, target_class, base_outname.replace('.png', '_resnet.png'))

    def get_shap_explainers(self, img : str, base_outname : str = './explainer_shap.png') -> None:
        '''
        Generate SHAP explainers for all four model for the provided input image.

        Parameters
        ----------
        img : str
            path to image to be explained
        base_outname : str
            sample output name and path to be modified to save all results. Must end
            in ".png"
        '''
        if base_outname[-4:] != '.png':
            raise ValueError(f'Keyword argument "base_outname" must end in ".png"')

        self.dnns.densenet.predict_with_shap(img, base_outname.replace('.png', '_densenet.png'))
        self.dnns.efficientnet.predict_with_shap(img, base_outname.replace('.png', '_efficientnet.png'))
        self.dnns.inception.predict_with_shap(img, base_outname.replace('.png', '_inception.png'))
        self.dnns.resnet.predict_with_shap(img, base_outname.replace('.png', '_resnet.png'))


    def get_lime_explainers(self, img : str, base_outname : str = './explainer_lime.png') -> None:
        '''
        Generate LIME explainers for all four model for the provided input image.

        Parameters
        ----------
        img : str
            path to image to be explained
        base_outname : str
            sample output name and path to be modified to save all results. Must end
            in ".png"
        '''
        if base_outname[-4:] != '.png':
            raise ValueError(f'Keyword argument "base_outname" must end in ".png"')

        self.dnns.densenet.predict_with_lime(img, base_outname.replace('.png', '_densenet.png'))
        self.dnns.efficientnet.predict_with_lime(img, base_outname.replace('.png', '_efficientnet.png'))
        self.dnns.inception.predict_with_lime(img, base_outname.replace('.png', '_inception.png'))
        self.dnns.resnet.predict_with_lime(img, base_outname.replace('.png', '_resnet.png'))


if __name__ == '__main__':

    #Example usage:
    model_paths = [ '../saved_models/dnns_mixup_3M/dense_net_w_hid_best_mixup_3M.pkl',
                    '../saved_models/dnns_mixup_3M/efficient_net_w_hid_best_mixup_3M.pkl',
                    '../saved_models/dnns_mixup_3M/inception_w_hid_best_mixup_3M.pkl',
                    '../saved_models/dnns_mixup_3M/res_net_w_hid_best_mixup_3M.pkl',]

    explainers = MediraiDNNExplainers(model_paths)
    
    sample_img, sample_is_malig = './sample_data/sample_img.jpg', 0

    explainers.get_all_explainers(sample_img, sample_is_malig, base_outname = './explainability_examples/explainer.png')



        
