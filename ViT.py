from typing import List
import torch 
import torch.nn as nn 

class PatchEmbed(nn.Module):
    """Split image into patches and then embed them.

    Parameters
    ----------
    img_size : int 
        size of the image
    
    patch_size : int 
        size of the patch (sq.)

    in_chan : int
        num of channels (1 / 3) 

    embed_dim : int 
        embeddgin dims to the transformers input 

    """

    def __init__(self, img_size, patch_size, in_chan=3, embed_size=728) -> None:
        super().__init__()
        self.img_size = img_size 
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2 

        self.proj = nn.Conv2d(
            in_chan, 
            embed_size, 
            kernel_size = patch_size, 
            stride = patch_size
        )

    def forward(self, x):
        """
        Run forward pass 
        
        input :
            shape ( n_samples, in_chans, img_size, img_size) 

        returns : 
            shape (n_samples, n_patches, embed_dims) 
        """
        x = self.proj(x) ## shape (n_samples, embed_size,  patch_size, patch_size ) 
        x = x.flatten(2)    ## args defines from which position to flatten the rest.; shape (n_samples, embed_dims, n_patches)
        x = x.transpose(1, 2) ## shape (n_samples, n_patches, embed_dims)

        return x

class Attention(nn.Module):
    """
    Attention mechanism : input and output dims are same 

    parameters 
    -----------
    dims : in 
        the input and output dims of per token fetures 

    n_heads : int 
        no. of attention heads 

    qkv_bias : bool 
        if True, include a bias to each query, key , value projections 

    attn_p : float 
        dropout prob applied to q, k, v 

    proj_p : float 
        dropout prob applied to output tensor 

    Attributes 
    -------------
    scale : float 
        normalizing dot product 

    qkv : nn.Linear 

    proj : nn.Linear (MLP)
        linear mapping that takes concatenated output of all attention heads and maps it into a new space 

    attn_drop, proj_drop : nn.Dropout 
        dropout layer 
    """

    def __init__(self, dims, n_heads=12, qkv_bias=True, attn_p = 0.0, proj_p = 0.0) -> None:
        super().__init__()
        self.n_heads= n_heads 
        self.dims = dims 
        self.head_dims = dims // n_heads
        self.scale = self.head_dims ** -0.5 ## idea from Attn is all you need, softmax should not take large values -> lower gradietns 

        self.qkv = nn.Linear(dims, dims*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p) 
        self.proj = nn.Linear(dims, dims) 
        self.proj_p = nn.Dropout(proj_p) 

    def forward(self, x):
        """
        Parameters 
        -----------
        x : torch.tensor 
            shape (n_samples, n_patches + 1 , dim) ## plus 1 dims cuz first dim is for [class] token
        
        Returns
        -----------
        torch.tensor 
            shape (n_samples, n_patches + 1 , dim)
        """

        n_samples, n_tokens, dims = x.shape ## token == n_patches (+ 1, [class] token, added later)
        if dims != self.dims:
            raise ValueError
        qkv = self.qkv(x) ## shape : (n_samples, n_patches + 1, n_dims * 3) 
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dims
        )
        qkv = qkv.permute(2, 0, 3, 1, 4) ## shape : (3, n_samples, n_heads, n_patches + 1, head_dims)
        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1) ## shape : (n_samples, n_heads, head_dims, n_patches + 1) ; since keys is to be multiplied (matmul) with queries 
        dp = (
            q @ k_t         ## @ - matmul ; this will execute since last two dims of q & k are aligned 
        ) ** self.scale ## shape : (n_samples, n_heads, n_patches + 1, n_patches + 1) 
        attn = dp.softmax(dim=1) 
        attn = self.attn_drop(attn)
        weighted_avg = attn @ v ## shape : (n_samples, n_heads, n_patches + 1, head_dims)
        weighted_avg = weighted_avg.transpose(1, 2) ## shape : (n_samples, n_patches + 1, n_heads, head_dims)

        x = self.proj(weighted_avg) 
        x = self.proj_p(x) 

        return x


class MLP(nn.Module):
    """
    Paramters 
    ---------
    input : torch.tensor 
        projections 
    return : int 
        num of out feat 
    """

    def __init__(self, in_feat, hidden_feat, out_feat, p=0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_feat, hidden_feat) 
        self.act = nn.GELU() ## gaussian activation unit instead of RELU 
        self.fc2 = nn.Linear(hidden_feat, out_feat) 
        self.drop = nn.Dropout(p) 

    def forward(self, x):
        x = self.act(self.fc1(x)) 
        x = self.drop(x) 
        x = self.fc2(x)
        x = self.drop(x) 

        return x 

class Block(nn.Module):
    """
    Transformer block 

    Parameters
    -------------
    dim : int 
        embedding dims 
    n_heads : int 
        num of attention heads
    mlp_ratio : float 
        determines the hidden dims size of 'MLP' module wrt 'dim' 
    qkv bias : bool 
        if true, bias for query, key and value projections 
    p, attn_p : float 
        dropout probability 

    Attributes 
    -------------
    norm1, norm2 : layernorm 
        layer normalization 
    attn : Attention 
        attention omdule 
    mlp : MLP 
        mlp module 

    """

    def __init__(self, dims, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dims, eps=1e-6)
        self.attn = Attention(
            dims, 
            n_heads, 
            qkv_bias, 
            attn_p, 
            p
        )
        self.norm2 = nn.LayerNorm(dims, eps=1e-6) 
        hidden_feat = int(dims ** mlp_ratio)
        self.MLP = MLP(
            dims, 
            hidden_feat=hidden_feat,
            out_feat=dims
        )

    def forward(self, x):
        """
        parameters:
        --------------
        x : torch.tensor
            shape : (n_samples, n_patches + 1, dims)

        returns : torch.tensor
            shape : (n_samples, n_patches + 1, dims) 
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):
    """
    Vision transformers 

    Parameters:
    ----------
    img_size : int
    patch_size : int 
    in_chans : int 
    n_classes : int
        num of classes for the output 
    embed_dims : int 
    depth : int 
        number for transformer blocks 
    n_heads : int 
    mlp_ratio : float 
    qkv_bias : bool 
    p, attn_p : float 

    Attributes 
    -------------
    patch_embed : PatchEmbed 
        isntance of 'PatchEmbed' layer 
    cls_token : nn.Parameter
        learnable parameter that will represent the first token in the sequence. it has 'embed_dim' elements 

    pos_emb : nn.Parameter
        position embeddings of cls_toek + all other patches 
        it has (n_patches + 1) * embed_dims elements 

    pos_drop : nn.Dropout 
    blocks : nn.ModuleList 
        list of 'Block' module 

    norm : nn.LayerNorm 
    """
    def __init__(self, img_size, patch_size, in_chan, n_classes, dims, depth, n_heads, mlp_ratio, qkv_bias=True, p=0., attn_p=0.) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chan, embed_size=dims)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1+self.patch_embed.n_patches, dims) 
        )
        self.pos_drop = nn.Dropout(p) 

        self.blocks = nn.ModuleList(
            [
                Block(
                    dims, 
                    n_heads, 
                    mlp_ratio, 
                    p, 
                    attn_p
                )
                for _ in range(depth)
            ]
        )
        self.norm1 = nn.LayerNorm(dims, eps=1e-6)
        self.head = nn.Linear(dims, n_classes) 


    def forward(self, x):
        """
        input: torch.tensor 
            batch of images ; shape : (n_samples, in_chan, img_size, img_size)

        returns : 
        logits : torch.tensor 
            logits over all the classes 
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x) 

        cls_token = self.cls_token.expand(
            n_samples, -1, -1       ## replicate the class token for all sets of the images ; shape : (n_samples, 1, embed_dims)
        )
        x = torch.cat((cls_token, x), dim=1) ## shape : (n_samples, 1 + n_patches, embed_dims) ; prepend the cls token to patch embeddings
        x = x + self.pos_embed ## add positional embeddgins to it 
        x = self.pos_drop(x) 

        for block in self.blocks:
            x = block(x) 

        x = self.norm1(x) 

        cls_token_final = x[:, 0] ## just the cls token; assumed that the cls token has all the information and discard the rest of the token 
        x = self.head(cls_token_final) 

        return x 

        





    