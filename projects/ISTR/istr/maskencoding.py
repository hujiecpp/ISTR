import torch
import torch.nn as nn
import torch_dct
import numpy as np


VALUE_MAX = 0.05
VALUE_MIN = 0.01

# -------- DCT --------
class DctMaskEncoding(object):
    """
    Apply DCT to encode the binary mask, and use the encoded vector as mask representation in instance segmentation.
    """
    def __init__(self, vec_dim, mask_size=128):
        """
        vec_dim: the dimension of the encoded vector, int
        mask_size: the resolution of the initial binary mask representaiton.
        """
        self.vec_dim = vec_dim
        self.mask_size = mask_size
        assert vec_dim <= mask_size*mask_size
        self.dct_vector_coords = self.get_dct_vector_coords(r=mask_size)

    def encode(self, masks):
        """
        Encode the mask to vector of vec_dim or specific dimention.
        """
        # if dim is None:
        dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        # else:
            # dct_vector_coords = self.dct_vector_coords[:dim]
        masks = masks.view([-1, self.mask_size, self.mask_size])#.to(dtype=float)  # [N, H, W]
        dct_all = torch_dct.dct_2d(masks, norm='ortho')
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_vectors = dct_all[:, xs, ys]  # reshape as vector
        return dct_vectors  # [N, D]

    def decode(self, dct_vectors, roi_feat=None):
        """
        intput: dct_vector numpy [N,dct_dim]
        output: mask_rc mask reconstructed [N, mask_size, mask_size]
        """
        device = dct_vectors.device
        # if dim is None:
        dct_vector_coords = self.dct_vector_coords[:self.vec_dim]
        # else:
        #     dct_vector_coords = self.dct_vector_coords[:dim]
        #     dct_vectors = dct_vectors[:, :dim]

        N = dct_vectors.shape[0]
        dct_trans = torch.zeros([N, self.mask_size, self.mask_size], dtype=dct_vectors.dtype).to(device)
        xs, ys = dct_vector_coords[:, 0], dct_vector_coords[:, 1]
        dct_trans[:, xs, ys] = dct_vectors
        mask_rc = torch_dct.idct_2d(dct_trans, norm='ortho')  # [N, mask_size, mask_size]
        return mask_rc

    def get_dct_vector_coords(self, r=128):
        """
        Get the coordinates with zigzag order.
        """
        dct_index = []
        for i in range(r):
            if i % 2 == 0:  # start with even number
                index = [(i-j, j) for j in range(i+1)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i+1)]
                dct_index.extend(index)
        for i in range(r, 2*r-1):
            if i % 2 == 0:
                index = [(i-j, j) for j in range(i-r+1,r)]
                dct_index.extend(index)
            else:
                index = [(j, i-j) for j in range(i-r+1,r)]
                dct_index.extend(index)
        dct_idxs = np.asarray(dct_index)
        return dct_idxs


# -------- PCA --------
@torch.no_grad()
class PCAMaskEncoding(nn.Module):
    """
    To do the mask encoding of PCA.
        components_: (tensor), shape (n_components, n_features) if agnostic=True
                                else (n_samples, n_components, n_features)
        explained_variance_: Variance explained by each of the selected components.
                            (tensor), shape (n_components) if agnostic=True
                                        else (n_samples, n_components)
        mean_: (tensor), shape (n_features) if agnostic=True
                          else (n_samples, n_features)
        agnostic: (bool), whether class_agnostic or class_specific.
        whiten : (bool), optional
        When True (False by default) the ``components_`` vectors are divided
        by ``n_samples`` times ``components_`` to ensure uncorrelated outputs
        with unit component-wise variances.
        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometimes
        improve the predictive accuracy of the downstream estimators by
        making data respect some hard-wired assumptions.
        sigmoid: (bool) whether to apply inverse sigmoid before transform.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.agnostic = True #cfg.MODEL.ISTR.AGNOSTIC
        self.whiten = True #cfg.MODEL.ISTR.WHITEN
        self.sigmoid = True #cfg.MODEL.ISTR.SIGMOID
        self.mask_feat_dim = cfg.MODEL.ISTR.MASK_FEAT_DIM
        self.mask_size = cfg.MODEL.ISTR.MASK_SIZE

        if self.agnostic:
            self.components = nn.Parameter(torch.zeros(self.mask_feat_dim, self.mask_size**2), requires_grad=False)
            self.explained_variances = nn.Parameter(torch.zeros(self.mask_feat_dim), requires_grad=False)
            self.means = nn.Parameter(torch.zeros(self.mask_size**2), requires_grad=False)
        else:
            raise NotImplementedError

    def inverse_sigmoid(self, x):
        """Apply the inverse sigmoid operation.
                y = -ln(1-x/x)
        """
        # In case of overflow
        value_random = VALUE_MAX * torch.rand_like(x)
        value_random = torch.where(value_random > VALUE_MIN, value_random, VALUE_MIN * torch.ones_like(x))
        x = torch.where(x > value_random, 1 - value_random, value_random)
        # inverse sigmoid
        y = -1 * torch.log((1 - x) / x)
        return y

    def encoder(self, X):
        """Apply dimensionality reduction to X.
        X is projected on the first principal components previously extracted
        from a training set.
        Parameters
        ----------
        X : Original features(tensor), shape (n_samples, n_features)
            New data, where n_samples is the number of samples
            and n_features is the number of features.
        Returns
        -------
        X_transformed : Transformed features(tensor), shape (n_samples, n_features)
        """
        X = X.flatten(1)
        assert X.shape[1] == self.mask_size**2, print("The original mask_size of input should be equal to the supposed size.")

        if self.sigmoid:
            X = self.inverse_sigmoid(X)

        if self.agnostic:
            if self.means is not None:
                X_transformed = X - self.means
            X_transformed = torch.matmul(X_transformed, self.components.T)
            if self.whiten:
                X_transformed /= torch.sqrt(self.explained_variances)
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        return X_transformed

    def decoder(self, X, roi_feat=None):
        """Transform data back to its original space.
        In other words, return an input X_original whose transform would be X.
        Parameters
        ----------
        X : Encoded features(tensor), shape (n_samples, n_components)
            New data, where n_samples is the number of samples
            and n_components is the number of components.
        Returns
        -------
        X_original original features(tensor), shape (n_samples, n_features)
        """
        X = X.flatten(1)
        assert X.shape[1] == self.mask_feat_dim, print("The dim of transformed data should be equal to the supposed dim.")

        if self.agnostic:
            if self.whiten:
                components_ = self.components * torch.sqrt(self.explained_variances.unsqueeze(1))
            X_transformed = torch.matmul(X, components_)
            if self.means is not None:
                X_transformed = X_transformed + self.means
        else:
            # TODO: The class-specific version has not implemented.
            raise NotImplementedError

        if self.sigmoid:
            X_transformed = torch.sigmoid(X_transformed)
        else:
            X_transformed = torch.clamp(X_transformed, min=0.01, max=0.99)

        return X_transformed


# -------- AE --------
def frozen(module):
    if getattr(module,'module',False):
        for child in module.module():
            for param in child.parameters():
                param.requires_grad = False
    else:
        for param in module.parameters():
            param.requires_grad = False

            
class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class up_conv(nn.Module):
    """
    Up Convolution Block
    """
    def __init__(self, in_ch, out_ch, size, scale_factor):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(size, scale_factor, mode='bilinear', align_corners=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Encoder(nn.Module):
    def __init__(self, mask_size, embedding_size):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1),  # 56
            nn.BatchNorm2d(16),
            nn.ELU(True),
            nn.Conv2d(16, 32, 4, 2, 1), # 28
            nn.BatchNorm2d(32),
            nn.ELU(True),
            nn.Conv2d(32, 64, 4, 2, 1), # 14
            nn.BatchNorm2d(64),
            nn.ELU(True),
            nn.Conv2d(64, 128, 4, 2, 1), # 7
            nn.BatchNorm2d(128),
            nn.ELU(True),

            nn.Conv2d(128, embedding_size, 7, 1),
            View((-1, embedding_size*1*1)),
        )
    
    def forward(self, x):
        f = self.encoder(x)
        return f


class Decoder(nn.Module):
    def __init__(self, mask_size, embedding_size):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            View((-1, embedding_size, 1, 1)),
            nn.ConvTranspose2d(embedding_size, 128, 7, 1),
            nn.BatchNorm2d(128),
            nn.ELU(inplace=True),
            up_conv(128, 64, None, 2),
            up_conv(64, 32, None, 2),
            up_conv(32, 16, None, 2),
            up_conv(16, 16, None, 2),
            nn.Conv2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
            View((-1, 1, mask_size, mask_size)),
        )

    def forward(self, x, roi_feat=None):
        for i, layer in enumerate(self.decoder):
            if i == 6 and roi_feat != None:
                shape = x.shape
                roi_feat = roi_feat.view(shape[0], shape[1], shape[2], shape[3])
                x = x + roi_feat
                del roi_feat
            x = layer(x)
        # x_rec = self.decoder(f)
        return x