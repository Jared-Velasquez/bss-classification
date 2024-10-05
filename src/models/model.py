import segmentation_models as sm

# Use Qubvel's Segmentation Models library to create Keras U-Net models with varying backbones.

# Encoder Weights
ENCODER_WEIGHTS = 'imagenet'

# Backbones
RESNET_50_BACKBONE = 'resnet50'
RESNET_101_BACKBONE = 'resnet101'
RESNET_152_BACKBONE = 'resnet152'

VGG_19_BACKBONE = 'vgg19'

INCEPTION_BACKBONE = 'inceptionv3'

# Optimizers
ADAM_OPTIMIZER = 'adam'
SGD_OPTIMIZER = 'sgd'

# Loss functions
JACCARD_LOSS = sm.losses.JaccardLoss

# Metrics
IOU_METRIC = sm.metrics.IOUScore
F1_METRIC = sm.metrics.FScore

# Activation functions
SIGMOID_ACTIVATION = 'sigmoid'
SOFTMAX_ACTIVATION = 'softmax'

def unet_backbone(
        backbone: str = RESNET_50_BACKBONE,
        classes: int = 1, 
        input_shape = (128, 128, 3), 
        activation: str = SIGMOID_ACTIVATION
        ):
    
    return sm.Unet(
        backbone_name=backbone, 
        input_shape=input_shape, 
        classes=classes, 
        activation=activation, 
        encoder_weights=ENCODER_WEIGHTS
        )