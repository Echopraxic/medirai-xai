from .dnns import ResNet50Hidden, ResNet18Hidden, WideResNet50Hidden
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
    arch = net_config["architecture"].upper()
    pretrained = "PRETRAINED" in arch

    if "RESNET18P" in arch:
        # was WideResNet50Hidden: every "resnet18p_pretrained" run was really a Wide-ResNet50 (CODEBASE_TODO P0-12)
        return ResNet18Hidden(num_classes=net_config["output_size"]) if pretrained             else resnet18(weights=None, num_classes=net_config["output_size"])

    if "RESNET50P" in arch:
        if "WIDE" in arch:
            return WideResNet50Hidden(num_classes=net_config["output_size"]) if pretrained                 else wide_resnet50_2(weights=None, num_classes=net_config["output_size"])
        return ResNet50Hidden(num_classes=net_config["output_size"]) if pretrained             else resnet50(weights=None, num_classes=net_config["output_size"])

    if arch == "UNET":
        return UNet(in_channels=net_config["in_channels"], output_size=net_config["output_size"],
                    feature_size=net_config["feature_size"], batch_norm=net_config["use_batchnorm"],
                    dropout=net_config["dropout"])

    if "WIDERESNET" in arch:
        depth, width = arch.split('_')[1:]
        return WideResNet(depth=int(depth), widen_factor=int(width), num_classes=net_config['output_size'],
                          dropout=net_config['dropout'])

    # (a bare `except` used to hide real construction errors behind this message)
    raise NotImplementedError(f"Architecture '{net_config['architecture']}' "
                              f"(network_type '{net_config['network_type']}') is not implemented in get_model()")
        
