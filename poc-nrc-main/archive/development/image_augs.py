
from torchvision import transforms, models

class ImageAugmentaitons:

    def __init__(self, img_size):

        self.img_size = img_size
    
    def train(self):

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),    
            transforms.RandomApply([
                transforms.RandomRotation(degrees=30)
                ], p=0.5),
            transforms.ColorJitter(
                    brightness=0.2,  
                    contrast=0.1,    
                    saturation=0.5,  
                    hue=0.1          
                ),
            transforms.EdgeEnhancementTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        return transform


    def val(self):

        transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        return transform


