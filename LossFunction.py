import torchvision
import torch

# Use GPU if available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get VGG19 model for reconstruction loss
model_vgg = torchvision.models.vgg19(pretrained = True).features.to(device).eval()

# freeze parameters
for param in model_vgg.parameters():
    param.requires_grad_(False)

# Define function that extreact features from layer
def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features
#end

#define final loss function
def LossFunction(x, y):
    with torch.no_grad():
        """refer to this link: https://velog.io/@jhpark/AutoEncoder-Reconstruction-%EA%B8%B0%EB%B0%98-Computer-Vision-%EC%9D%B4%EC%83%81%EC%B9%98-%ED%83%90%EC%A7%80-%EC%BD%94%EB%93%9C-%EB%A6%AC%EB%B7%B0 
        to find # of each layer in vgg19"""
        feature_x = get_features(x, model_vgg, {"26": "relu4_4"})["relu4_4"]
        feature_y = get_features(y, model_vgg, {"26": "relu4_4"})["relu4_4"]
    l1_loss_func = torch.nn.L1Loss()
    l1_loss = l1_loss_func(x, y)
    mse_loss_func = torch.nn.MSELoss()
    recon_loss = mse_loss_func(feature_x, feature_y)
    return l1_loss + recon_loss
#end