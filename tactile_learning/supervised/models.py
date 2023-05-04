import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from pytorch_model_summary import summary
from vit_pytorch.vit import ViT


def softbound(x, x_min, x_max):
    return (torch.log1p(torch.exp(-torch.abs(x - x_min)))  \
        - torch.log1p(torch.exp(-torch.abs(x - x_max)))) \
        + torch.maximum(x, x_min) + torch.minimum(x, x_max) - x


def create_model(
    in_dim,
    in_channels,
    out_dim,
    model_params,
    saved_model_dir=None,
    display_model=True,
    device='cpu'
):

    if 'mdn_kwargs' in model_params:
        model_out_dim = model_params['mdn_kwargs']['model_out_dim']
    else:
        model_out_dim = out_dim

    if model_params['model_type'] in ['fcn']:
        model = FCN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] in ['simple_cnn', 'posenet_cnn']:
        model = CNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=model_out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'nature_cnn':
        model = NatureCNN(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=model_out_dim,
            **model_params['model_kwargs']
        ).to(device)
        model.apply(weights_init_normal)

    elif model_params['model_type'] == 'resnet':
        model = ResNet(
            ResidualBlock,
            in_channels=in_channels,
            out_dim=model_out_dim,
            **model_params['model_kwargs'],
        ).to(device)

    elif model_params['model_type'] == 'vit':
        model = ViT(
            image_size=in_dim[0],
            channels=in_channels,
            num_classes=model_out_dim,
            **model_params['model_kwargs']
        ).to(device)

    elif model_params['model_type'] in ['cnn_mdn_jl', 'cnn_mdn_jl_pretrain']:
        model = MDN_JL(
            in_dim=in_dim,
            in_channels=in_channels,
            out_dim=model_out_dim,
            **model_params['model_kwargs']
        ).to(device)

    else:
        raise ValueError('Incorrect model_type specified:  %s' % (model_params['model_type'],))

    if 'mdn_kwargs' in model_params:
        model = MDNHead(
            model=model,
            out_dim=out_dim,
            **model_params['mdn_kwargs']
        ).to(device)

    if saved_model_dir is not None:
        model.load_state_dict(torch.load(os.path.join(
            saved_model_dir, 'best_model.pth'), map_location='cpu')
        )

    if display_model:
        if model_params['model_type'] == 'fcn':
            dummy_input = torch.zeros((1, in_dim)).to(device)
        else:
            dummy_input = torch.zeros((1, in_channels, *in_dim)).to(device)
        print(summary(
            model,
            dummy_input,
            show_input=True
        ))

    return model


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class FCN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        activation='relu',
        dropout=0.0,
        apply_batchnorm=False,
    ):
        super(FCN, self).__init__()

        assert len(fc_layers) > 0, "fc_layers must contain values"

        fc_modules = []

        # add first layer
        fc_modules.append(nn.Linear(in_dim, fc_layers[0]))
        if apply_batchnorm:
            fc_modules.append(nn.BatchNorm1d(fc_layers[0]))
        if activation == 'relu':
            fc_modules.append(nn.ReLU())
        elif activation == 'elu':
            fc_modules.append(nn.ELU())

        # add remaining layers
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if apply_batchnorm:
                fc_modules.append(nn.BatchNorm1d(fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        return self.fc(x)


class CNN(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_layers=[16, 16, 16],
        conv_kernel_sizes=[5, 5, 5],
        fc_layers=[128, 128],
        activation='relu',
        apply_batchnorm=False,
        dropout=0.0,
    ):
        super(CNN, self).__init__()

        assert len(conv_layers) > 0, "conv_layers must contain values"
        assert len(fc_layers) > 0, "fc_layers must contain values"
        assert len(conv_layers) == len(conv_kernel_sizes), "conv_layers must be same len as conv_kernel_sizes"

        # add first layer to network
        cnn_modules = []
        cnn_modules.append(nn.Conv2d(in_channels, conv_layers[0], kernel_size=conv_kernel_sizes[0], stride=1, padding=2))
        if apply_batchnorm:
            cnn_modules.append(nn.BatchNorm2d(conv_layers[0]))
        if activation == 'relu':
            cnn_modules.append(nn.ReLU())
        elif activation == 'elu':
            cnn_modules.append(nn.ELU())
        cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # add the remaining conv layers by iterating through params
        for idx in range(len(conv_layers) - 1):
            cnn_modules.append(
                nn.Conv2d(
                    conv_layers[idx],
                    conv_layers[idx + 1],
                    kernel_size=conv_kernel_sizes[idx + 1],
                    stride=1, padding=2)
                )

            if apply_batchnorm:
                cnn_modules.append(nn.BatchNorm2d(conv_layers[idx+1]))

            if activation == 'relu':
                cnn_modules.append(nn.ReLU())
            elif activation == 'elu':
                cnn_modules.append(nn.ELU())
            cnn_modules.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

        # create cnn component of network
        self.cnn = nn.Sequential(*cnn_modules)

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        if activation == 'relu':
            fc_modules.append(nn.ReLU())
        elif activation == 'elu':
            fc_modules.append(nn.ELU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            if activation == 'relu':
                fc_modules.append(nn.ReLU())
            elif activation == 'elu':
                fc_modules.append(nn.ELU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class NatureCNN(nn.Module):
    """
    CNN from DQN Nature paper (Commonly used in RL):
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.
    """

    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        fc_layers=[128, 128],
        dropout=0.0
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # compute shape out of cnn by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            n_flatten = np.prod(self.cnn(dummy_input).shape)

        fc_modules = []
        fc_modules.append(nn.Linear(n_flatten, fc_layers[0]))
        fc_modules.append(nn.ReLU())
        for idx in range(len(fc_layers) - 1):
            fc_modules.append(nn.Linear(fc_layers[idx], fc_layers[idx + 1]))
            fc_modules.append(nn.ReLU())
            fc_modules.append(nn.Dropout(dropout))
        fc_modules.append(nn.Linear(fc_layers[-1], out_dim))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x):
        x = self.cnn(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, in_channels, layers, out_dim):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512, out_dim)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class MDNHead(nn.Module):
    """
    Implementation of Mixture Density Networks in Pytorch from

    https://github.com/tonyduan/mixture-density-network

    Mixture density network.
    [ Bishop, 1994 ]
    Parameters
    ----------
    dim_in: int; dimensionality of the covariates
    dim_out: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """

    def __init__(
        self,
        model,
        out_dim,
        model_out_dim,
        hidden_dims,
        activation,
        n_mdn_components,
        noise_type='diagonal',
        fixed_noise_level=None
    ):
        super(MDNHead, self).__init__()

        assert (fixed_noise_level is not None) == (noise_type == 'fixed')

        num_sigma_channels = {
            'diagonal': out_dim * n_mdn_components,
            'isotropic': n_mdn_components,
            'isotropic_across_clusters': 1,
            'fixed': 0,
        }[noise_type]

        self.out_dim, self.n_components = out_dim, n_mdn_components
        self.noise_type, self.fixed_noise_level = noise_type, fixed_noise_level

        # init pi and normal heads
        pi_network_modules = [model]
        normal_network_modules = [model]

        # add the first layer
        pi_network_modules.append(nn.ReLU())
        pi_network_modules.append(nn.Linear(model_out_dim, hidden_dims[0]))
        normal_network_modules.append(nn.ReLU())
        normal_network_modules.append(nn.Linear(model_out_dim, hidden_dims[0]))
        if activation == 'relu':
            pi_network_modules.append(nn.ReLU())
            normal_network_modules.append(nn.ReLU())
        elif activation == 'elu':
            pi_network_modules.append(nn.ELU())
            normal_network_modules.append(nn.ELU())

        # add the remaining hidden layers
        for idx in range(len(hidden_dims) - 1):
            pi_network_modules.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            normal_network_modules.append(nn.Linear(hidden_dims[idx], hidden_dims[idx + 1]))
            if activation == 'relu':
                pi_network_modules.append(nn.ReLU())
                normal_network_modules.append(nn.ReLU())
            elif activation == 'elu':
                pi_network_modules.append(nn.ELU())
                normal_network_modules.append(nn.ELU())

        # add the final layers
        pi_network_modules.append(nn.Linear(hidden_dims[-1], n_mdn_components))
        normal_network_modules.append(nn.Linear(hidden_dims[-1], out_dim * n_mdn_components + num_sigma_channels))

        self.pi_network = nn.Sequential(*pi_network_modules)
        self.normal_network = nn.Sequential(*normal_network_modules)

        # self.pi_network = nn.Sequential(
        #     model,
        #     nn.ReLU(),
        #     nn.Linear(out_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, n_mdn_components),
        # )
        #
        # self.normal_network = nn.Sequential(
        #     model,
        #     nn.ReLU(),
        #     nn.Linear(out_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, out_dim * n_mdn_components + num_sigma_channels)
        # )

    def forward(self, x, eps=1e-6):
        """
        Returns
        -------
        log_pi: (bsz, n_components)
        mu: (bsz, n_components, dim_out)
        sigma: (bsz, n_components, dim_out)
        """

        log_pi = torch.log_softmax(self.pi_network(x), dim=-1)
        normal_params = self.normal_network(x)
        mu = normal_params[..., :self.out_dim * self.n_components]
        sigma = normal_params[..., self.out_dim * self.n_components:]

        if self.noise_type == 'diagonal':
            sigma = torch.exp(sigma + eps)
        if self.noise_type == 'isotropic':
            sigma = torch.exp(sigma + eps).repeat(1, self.dim_out)
        if self.noise_type == 'isotropic_across_clusters':
            sigma = torch.exp(sigma + eps).repeat(1, self.n_components * self.dim_out)
        if self.noise_type == 'fixed':
            sigma = torch.full_like(mu, fill_value=self.fixed_noise_level)

        mu = mu.reshape(-1, self.n_components, self.out_dim)
        sigma = sigma.reshape(-1, self.n_components, self.out_dim)

        return log_pi, mu, sigma

    def loss(self, x, y):
        """
        Calculates negative log_likelihood.
        """
        log_pi, mu, sigma = self.forward(x)
        z_score = (y.unsqueeze(1) - mu) / sigma
        normal_loglik = (
            -0.5 * torch.einsum("bij,bij->bi", z_score, z_score)
            - torch.sum(torch.log(sigma), dim=-1)
        )
        loglik = torch.logsumexp(log_pi + normal_loglik, dim=-1)
        return -loglik

    def predict(self, x, deterministic=True):
        """
        Samples from the predicted distribution.
        """
        log_pi, mu, sigma = self.forward(x)
        pi = torch.exp(log_pi)
        pred_mean = torch.sum(pi.unsqueeze(dim=-1) * mu, dim=1)
        pred_stddev = torch.sqrt(torch.sum(pi.unsqueeze(dim=-1) * (sigma**2 + mu**2), dim=1) - pred_mean**2).squeeze()
        if deterministic:
            return pred_mean, pred_stddev
        else:
            pi_distribution = Normal(pred_mean, pred_stddev)
            return pi_distribution.rsample()

class MDN_JL(nn.Module):
    def __init__(
        self,
        in_dim,
        in_channels,
        out_dim,
        conv_filters,
        conv_kernel_sizes,
        conv_padding,
        conv_batch_norm,
        conv_activation,
        conv_pool_size,
        fc_units,
        fc_activation,
        fc_dropout,
        mix_components,
        pi_dropout,
        mu_dropout,
        sigma_inv_dropout,
        mu_min,
        mu_max,
        sigma_inv_min,
        sigma_inv_max,
    ):
        super(MDN_JL, self).__init__()

        self.in_dim, self.in_channels, self.out_dim, self.mix_components = \
            torch.tensor(in_dim), torch.tensor(in_channels), torch.tensor(out_dim), torch.tensor(mix_components)
        self.mu_min, self.mu_max, self.sigma_inv_min, self.sigma_inv_max = \
            torch.tensor(mu_min), torch.tensor(mu_max), torch.tensor(sigma_inv_min), torch.tensor(sigma_inv_max)

        activ_modules = {'relu': nn.ReLU, 'elu': nn.ELU}

        # convolutional base
        conv_modules = []
        for i in range(len(conv_filters)):
            conv_modules.append(
                nn.Conv2d(
                    in_channels=in_channels if i == 0 else conv_filters[i - 1],
                    out_channels=conv_filters[i],
                    kernel_size=conv_kernel_sizes[i],
                    padding=conv_padding)
            )
            if conv_batch_norm:
                conv_modules.append(nn.BatchNorm2d(conv_filters[i]))
            conv_modules.append(activ_modules[conv_activation]())
            conv_modules.append(nn.MaxPool2d(kernel_size=conv_pool_size, stride=conv_pool_size))
        self.conv_base = nn.Sequential(*conv_modules)

        # compute number of conv base outputs by doing one forward pass
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, *in_dim))
            conv_base_outputs = np.prod(self.conv_base(dummy_input).shape)

        # shared fc layers
        fc_modules = []
        for i in range(len(fc_units)):
            if fc_dropout > 0:
                fc_modules.append(nn.Dropout(fc_dropout))
            fc_modules.append(nn.Linear(conv_base_outputs if i == 0 else fc_units[i - 1], fc_units[i]))
            fc_modules.append(activ_modules[fc_activation]())
        self.fc_hidden = nn.Sequential(*fc_modules)

        fc_hidden_outputs = fc_units[-1] if len(fc_units) > 0 else conv_base_outputs

        # mixture weights
        pi_modules = []
        pi_modules.append(nn.Dropout(pi_dropout))
        pi_modules.append(nn.Linear(fc_hidden_outputs, mix_components))
        pi_modules.append(nn.Softmax(dim=-1))
        self.pi_head = nn.Sequential(*pi_modules)

        # component means and (inverse) stdevs
        self.mu_heads, self.sigma_inv_heads = [], []
        for i in range(out_dim):
            mu_modules_i = []
            mu_modules_i.append(nn.Dropout(mu_dropout[i]))
            mu_modules_i.append(nn.Linear(fc_hidden_outputs, mix_components))
            self.mu_heads.append(nn.Sequential(*mu_modules_i))

            sigma_inv_modules_i = []
            sigma_inv_modules_i.append(nn.Dropout(sigma_inv_dropout[i]))
            sigma_inv_modules_i.append(nn.Linear(fc_hidden_outputs, mix_components))
            self.sigma_inv_heads.append(nn.Sequential(*sigma_inv_modules_i))
        self.mu_heads, self.sigma_inv_heads = nn.ModuleList(self.mu_heads), nn.ModuleList(self.sigma_inv_heads)

    def _apply(self, fn):
        super(MDN_JL, self)._apply(fn)
        self.in_dim, self.in_channels, self.out_dim, self.mix_components = \
            fn(self.in_dim), fn(self.in_channels), fn(self.out_dim), fn(self.mix_components)
        self.mu_min, self.mu_max, self.sigma_inv_min, self.sigma_inv_max = \
            fn(self.mu_min), fn(self.mu_max), fn(self.sigma_inv_min), fn(self.sigma_inv_max)
        return self

    def forward(self, x):
        x = self.conv_base(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_hidden(x)
        pi = self.pi_head(x)
        mu, sigma_inv = [], []
        for i in range(self.out_dim):
            mu.append(softbound(self.mu_heads[i](x), self.mu_min[i], self.mu_max[i]))
            sigma_inv.append(softbound(self.sigma_inv_heads[i](x), self.sigma_inv_min[i], self.sigma_inv_max[i]))
        mu, sigma_inv = torch.stack(mu, dim=2), torch.stack(sigma_inv, dim=2)
        return pi, mu, sigma_inv

    def loss(self, x, y):
        pi, mu, sigma_inv = self.forward(x)
        squared_err = torch.sum(torch.square((torch.unsqueeze(y, dim=1) - mu) * sigma_inv), dim=2)
        log_pdf_comp = - (squared_err / 2) - (self.out_dim * np.log(2 * np.pi) / 2) \
                       + torch.sum(torch.log(sigma_inv), dim=2)
        log_pdf = torch.logsumexp(torch.log(pi) + log_pdf_comp, dim=-1)
        nll = -torch.mean(log_pdf)
        return nll

    def predict(self, x):
        pi, mu, sigma_inv = self.forward(x)
        pred_mean = torch.sum(pi.unsqueeze(dim=-1) * mu, dim=1)
        pred_stddev = torch.sqrt(torch.sum(pi.unsqueeze(dim=-1) * (1/(sigma_inv**2) + mu**2), dim=1)
                                 - pred_mean**2).squeeze()
        return pred_mean, pred_stddev
