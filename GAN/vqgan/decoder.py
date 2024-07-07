import torch
from torch import nn
from torch.nn import functional as F

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(VAE_AttentionBlock, self).__init__()
        self.in_channels = channels

        self.gn = nn.GroupNorm(32, channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A
    
class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_filters)
        self.conv_1 = nn.Conv2d(in_filters, out_filters, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_filters)
        self.conv_2 = nn.Conv2d(out_filters, out_filters, kernel_size=3, padding=1)

        if in_filters == out_filters:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_filters, out_filters, kernel_size=1, padding=0)
    
    def forward(self, x):
        residue = x

        x = self.groupnorm_1(x)
        x = F.silu(x) 
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self, args):
        super().__init__(
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
            nn.Conv2d(args.latent_dim, 512, kernel_size=1, padding=0),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512), 

            VAE_AttentionBlock(512), 

            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            # (Batch_Size, 512, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Upsample(scale_factor=2),
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 4, Width / 4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            # (Batch_Size, 512, Height / 4, Width / 4) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 512, Height / 2, Width / 2) -> (Batch_Size, 512, Height / 2, Width / 2)
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 

            VAE_ResidualBlock(512, 256), 
            VAE_ResidualBlock(256, 256), 
            VAE_ResidualBlock(256, 256), 
            
            # (Batch_Size, 256, Height / 2, Width / 2) -> (Batch_Size, 256, Height, Width)
            nn.Upsample(scale_factor=2), 
            
            # (Batch_Size, 256, Height, Width) -> (Batch_Size, 256, Height, Width)
            nn.Conv2d(256, 256, kernel_size=3, padding=1), 

            VAE_ResidualBlock(256, 128), 
            VAE_ResidualBlock(128, 128), 
            VAE_ResidualBlock(128, 128), 

            nn.GroupNorm(32, 128), 
            nn.SiLU(), 
            
            nn.Conv2d(128, args.image_channels, kernel_size=3, padding=1), 
        )

    def forward(self, x):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        
        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        # (Batch_Size, 3, Height, Width)
        return x    
    
        
        