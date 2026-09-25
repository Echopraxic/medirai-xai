from .dnns import ResNet50Hidden, WideResNet50Hidden
from .wideresnet import WideResNet
from .unet import UNet
from torchvision.models import resnet50, resnet18, wide_resnet50_2

def get_model(net_config) -> object:
    """
    Returns a model based on the provided configuration.
    Args:
        net_config (dict): Configuration dictionary for the network.
            configs: architecture, input_size, output_size, in_channels, feature_size, batchnorm, dropout
    Returns:
        object: An instance of the specified model.
    """
    try:           
        if "RESNET18P" in net_config["architecture"].upper():
            if "PRETRAINED" in net_config["architecture"].upper():
                net = WideResNet50Hidden(num_classes=net_config["output_size"])
            else: 
                net = resnet18(weights=None, num_classes=net_config["output_size"])

        elif "RESNET50P" in net_config["architecture"].upper():
            if "WIDE" in net_config["architecture"].upper():
                if "PRETRAINED" in net_config["architecture"].upper():
                    net = WideResNet50Hidden(num_classes=net_config["output_size"])
                else: 
                    net = wide_resnet50_2(weights=None, num_classes=net_config["output_size"])
            else: 
                if "PRETRAINED" in net_config["architecture"].upper():
                    net = ResNet50Hidden(num_classes=net_config["output_size"])
                else:
                    net = resnet50(weights=None, num_classes=net_config["output_size"])

        elif net_config["architecture"].upper() == "UNET":
            net =  UNet(in_channels=net_config["in_channels"], output_size=net_config["output_size"], 
                        feature_size=net_config["feature_size"], batch_norm=net_config["use_batchnorm"], 
                        dropout=net_config["dropout"])   
            
        elif 'WIDERESNET' in net_config["architecture"].upper():
                depth, width = net_config["architecture"].upper().split('_')[1:]
                net = WideResNet(depth=int(depth), widen_factor=int(width), num_classes=net_config['output_size'], dropout=net_config['dropout'])
            
        return net
    
    except: 
        net_type, net_arc = net_config["network_type"], net_config["architecture"]
        raise Exception(f"Network Type <<{net_type, net_arc}>> is not implemented in get_model()")
        
