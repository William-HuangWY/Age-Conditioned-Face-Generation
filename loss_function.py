import torch
import torch.nn.functional as F

#x = original, x_hat = prediction
def ELBO_loss(x_hat, x_real, mu, log_var):
    """
    x_hat: reconstructed image (B, 3, H, W)
    x: original image (B, 3, H, W)
    mu, log_var: encoder outputs
    """
    # Reconstruction loss (pixel-wise BCE or MSE)
    recon_loss = F.mse_loss(x_hat, x_real, reduction="mean")  # mean over batch and pixels
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kl_loss

def identity_loss(x_real, x_hat, identity_net):
    """
    x_real, x_hat: (B, C, H, W), normalized as required by identity_net
    identity_net: pre-trained network, output normalized embeddings
    """
    with torch.no_grad():  # pre-trained net
        feat_real = identity_net(x_real)  # (B, embedding_dim)
    feat_fake = identity_net(x_hat)      # (B, embedding_dim)
    
    # cosine similarity
    feat_real = F.normalize(feat_real, p=2, dim=1)
    feat_fake = F.normalize(feat_fake, p=2, dim=1)
    
    loss = 1 - (feat_real * feat_fake).sum(dim=1).mean()  # 1 - cosine similarity
    return loss

def age_loss(x_hat, age_label, age_net):
    """
    x_hat: generated image (B, C, H, W)
    age_label: one-hot age group labels (B, num_age_classes)
    age_net: pre-trained age classifier, output logits
    """
    logits = age_net(x_hat)  # (B, num_age_classes)
    loss = F.cross_entropy(logits, age_label)
    return loss