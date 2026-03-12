import torch
import torch.nn.functional as F
from face_age_dataset import AGE_BUCKETS


#x = original, x_hat = prediction
def ELBO_loss(x_hat, x_real, mu, log_var):
    """
    x_hat: reconstructed image (B, 3, H, W)
    x: original image (B, 3, H, W)
    mu, log_var: encoder outputs
    """
    # Reconstruction loss (pixel-wise BCE or MSE)
    recon_loss = F.l1_loss(x_hat, x_real, reduction="mean")  # mean over batch and pixels
    
    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss , kl_loss

def identity_loss(x_real, x_hat, identity_net):
    """
    x_real, x_hat: [B, C, H, W], float tensor, range [0,1] 或 [-1,1]
    identity_net: InceptionResnetV1
    """
    x_real = F.interpolate(x_real, size=(112, 112), mode='bilinear', align_corners=False)
    x_hat  = F.interpolate(x_hat,  size=(112, 112), mode='bilinear', align_corners=False)

    x_real = (x_real - 0.5) / 0.5
    x_hat  = (x_hat  - 0.5) / 0.5

    with torch.no_grad():
        feat_real = identity_net(x_real)
    feat_fake = identity_net(x_hat)

    feat_real = F.normalize(feat_real, dim=1)
    feat_fake = F.normalize(feat_fake, dim=1)

    cos_sim = (feat_real * feat_fake).sum(dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

    loss = 1 - cos_sim.mean()
    return loss

def age_loss(x_hat, age_label, age_net):
    """
    x_hat: generated image (B, C, H, W)
    age_label: one-hot age group labels (B, num_age_classes)
    age_net: pre-trained DEX age estimator
    """
    x = F.interpolate(x_hat, size=(224,224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1).to(x_hat.device)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1).to(x_hat.device)
    prob = age_net((x - mean) / std)  # (B,101)
    bucket_prob = torch.stack([prob[:, low:high+1].sum(dim=1) for low, high in AGE_BUCKETS], dim=1)
    loss = F.nll_loss(torch.log(bucket_prob + 1e-8), age_label.argmax(dim=1))
    return loss