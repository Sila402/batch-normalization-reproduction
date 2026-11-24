
import torch
import torch.nn as nn


class NoisyBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d followed by the time-varying noise of
    Santurkar et al. (2018), Appendix A.2.1 / Algorithm 1.

    At every training step, for every channel j:
        mu_j    ~ U(-n_mu, n_mu)
        sigma_j ~ U(1, n_sigma)
    then, for every sample and every spatial position
    (height and width are treated as batch dimensions):
        m ~ U(mu_j - r_mu, mu_j + r_mu)
        s ~ N(sigma_j, r_sigma)
    and the output becomes s * a + m.

    The noise is applied only in train mode. The paper does not say
    whether noise is used at test time; this is our assumption.
    """

    def __init__(self, num_features, n_mu=0.5, n_sigma=1.25,
                 r_mu=0.1, r_sigma=0.1):
        super().__init__(num_features)
        self.n_mu = n_mu
        self.n_sigma = n_sigma
        self.r_mu = r_mu
        self.r_sigma = r_sigma

    def forward(self, x):
        out = super().forward(x)

        if not self.training:
            return out

        channels = out.shape[1]
        shape = (1, channels, 1, 1)

        # Per-channel parameters, resampled at every step.
        mu = torch.empty(shape, device=out.device).uniform_(
            -self.n_mu, self.n_mu
        )
        sigma = torch.empty(shape, device=out.device).uniform_(
            1.0, self.n_sigma
        )

        # Per-activation noise around those parameters.
        m = mu + (torch.rand_like(out) * 2 - 1) * self.r_mu
        s = sigma + torch.randn_like(out) * self.r_sigma

        return s * out + m


class VGGA(nn.Module):
    def __init__(self, use_batchnorm=False, noisy_bn=False):
        super().__init__()

        if noisy_bn and not use_batchnorm:
            raise ValueError("noisy_bn requires use_batchnorm=True")

        # VGG11-based architecture for CIFAR-10
        configuration = [
            64, "M",
            128, "M",
            256, 256, "M",
            512, 512, "M",
            512, 512, "M"
        ]

        layers = []
        in_channels = 3

        for item in configuration:
            if item == "M":
                layers.append(nn.MaxPool2d(2))
                continue

            layers.append(
                nn.Conv2d(
                    in_channels,
                    item,
                    kernel_size=3,
                    padding=1,
                    bias=not use_batchnorm
                )
            )

            if use_batchnorm:
                if noisy_bn:
                    layers.append(NoisyBatchNorm2d(item))
                else:
                    layers.append(nn.BatchNorm2d(item))

            layers.append(nn.ReLU(inplace=True))
            in_channels = item

        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    for use_bn in [False, True]:
        model = VGGA(use_batchnorm=use_bn).to(device)
        model.eval()

        x = torch.randn(8, 3, 32, 32, device=device)

        with torch.no_grad():
            output = model(x)

        name = "With BN" if use_bn else "Without BN"

        print(name)
        print("Output shape:", output.shape)
        print(
            "Parameters:",
            sum(p.numel() for p in model.parameters())
        )

        del model, x, output

    print("VGG model test completed!")
