
import torch
import torch.nn as nn


class VGGA(nn.Module):
    def __init__(self, use_batchnorm=False):
        super().__init__()

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
