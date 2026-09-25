# MedirAI CNNs and Foundational Model Classifier

Authors:
 - Tyson Mitchell (tyson.mitchell@nrc-cnrc.gc.ca)
 - Erfan Fatehi (erfan.fatehi@nrc-cnrc.gc.ca)

Welcome the repo for MedirAI's skin lesion classifier. In its current state the model will predict a skin lesion as benign (0) or malignant (1). Model consists of four (4) deep-neural networks (DNNs), DenseNet, EfficientNet, Inception and ResNet. It also uses two (2) transformer based Foundational models: BiomedClip from Microsoft and DermFoundation from Google. At time of development both models were permissible to use from source.

# Overview
The six models used are mixed in a convoluted way, that may be difficult to explain. That being said, an overview is provided below.

Each of the model used as 3-layer MLP placed at the end of the model, meaning the output of the model (for each is a 1000+ component vector) is fed into a MLP that consists of hidden layer, and the final classification. For ResNet this would look like this:
```
input_img 
    -> ResNet 
        -> image_net_weights 
            -> MLP_input 
                -> MLP_hidden 
                    -> MLP_output 
                        -> classification
```
The same holds for the other models, although `image_net_weights` would be replaced with `image_embeddings` in the case of the Foundational Models. 

## Feature Fusion
To better leverage the feature representation, we deployed our own variety of feature fusion. Generally, this involves taking the feature representations found in `MLP_hidden` layers of one or more of the models, concatenating them and using this combined vector as the input to a new MLP network. In total this action in completed three times. There are (1) Feature Fusion across all DNNs (FF-DNN) and (2) Features Fusion across all Foundational Models (FF-FM). We then perform Feature Fusion across FF-DNN and FF-FM, which we refer as Global Feature Fusion (FF-G).

# Regarding the Code Base

## Interacting with the Models
The DNNs are sourced from `torchvision` and are wrapped in a helper class that inherit from model-agnostic class that contains most of the functionality. The DNN models can all be used simultaneously from the class `MediraiEnsembleModelV1`. This is the intended access points for the DNNs. Each foundational model have their own access points, `MediraiBiomedClip` and `MediraiDermFoundaion`. Each of these have example and started code in the `if __name__ == '__main__'` section at the end their corresponding files. 

## Making predictions
There is also an inference engine, which requires model be loaded only once to make inference calls at every level. This is a verbose return for debugging and deep-dive purposes.  Please see class `MediraiInferenceEngine`. As below, the `if __name__ == '__main__'` section at the bottom as example usage in the corresponding file.

## Explainability Plots
The ability to produce explainability plots also exists in class `MediraiDNNExplainers. Again with sample usage at the bottom of the corresponding files. This class will call GradCAM, SHAP, and LIME explainers. The source code has useful links for understanding what is happening in these graphics. Please note, that these explainers are correlation detectors and not causal explanations. 

# Important Remarks
- The version of ResNet provided by `torchvision` has limitations when it comes to use of the explainability tools. The exact details are unclear, but it appears to stem from certain blocks being reused on the backed that makes tracing gradients difficult. Thus, there is a drop in replacements round in `resent_no_resuse.py`.

