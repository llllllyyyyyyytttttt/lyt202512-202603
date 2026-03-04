from types_ import *
from torch import nn
from abc import abstractmethod

class BaseVAE(nn.Module):
    
    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass






from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')














import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X = data['data']
X = StandardScaler().fit_transform(X)
X_tensor = torch.tensor(X, dtype=torch.float32)

dataset = TensorDataset(X_tensor, X_tensor)  # VAE输入=输出
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

input_dim = X_tensor.shape[1]
latent_dim = 5
target_dim = input_dim

# 超参数
eta_dec_sq = 1.0
eta_prior_sq = 1.0
beta = 2.0  # beta系数控制KL权重

# -------------------------------
# 2. Linear Variational Encoder & Decoder
# -------------------------------
class LinearVariationalEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.nn_mean = nn.Linear(input_dim, latent_dim, bias=False)
        self.nn_logvar = nn.Linear(input_dim, latent_dim, bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        eps = torch.randn(batch_size, self.nn_mean.out_features, device=x.device)
        mu = self.nn_mean(x)
        logvar = self.nn_logvar(x)
        sigma = torch.exp(logvar / 2)
        z = mu + sigma * eps
        return {'z': z, 'mu': mu, 'sigma': sigma}

class LinearVariationalDecoder(nn.Module):
    def __init__(self, latent_dim, target_dim):
        super().__init__()
        self.l = nn.Linear(latent_dim, target_dim, bias=False)

    def forward(self, z):
        return self.l(z)

# -------------------------------
# 3. LinearBetaVAE
# -------------------------------
class LinearBetaVAE(nn.Module):
    def __init__(self, input_dim, latent_dim, target_dim, eta_dec_sq, eta_prior_sq, beta):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.eta_dec_sq = eta_dec_sq
        self.eta_prior_sq = eta_prior_sq
        self.beta = beta

        self.encoder = LinearVariationalEncoder(input_dim, latent_dim)
        self.decoder = LinearVariationalDecoder(latent_dim, target_dim)

    def forward(self, x, y):
        encoded = self.encoder(x)
        y_pred = self.decoder(encoded['z'])

        rec_loss = torch.square(y - y_pred).sum(-1).mean(0) / self.eta_dec_sq / 2

        kl_loss = 0.5 * (
            - torch.log(encoded['sigma']**2 / self.eta_prior_sq).sum(-1)
            - self.latent_dim
            + torch.norm(encoded['mu'], p=2, dim=-1)**2 / self.eta_prior_sq
            + (encoded['sigma']**2 / self.eta_prior_sq).sum(-1)
        ).mean(0)

        loss = rec_loss + self.beta * kl_loss

        return {
            'z': encoded['z'],
            'mu': encoded['mu'],
            'sigma': encoded['sigma'],
            'y_pred': y_pred,
            'loss': loss,
            'rec_loss': rec_loss,
            'kl_loss': kl_loss * self.beta,
            'enc_norm': self.encoder.nn_mean.weight.norm(),
            'dec_norm': self.decoder.l.weight.norm()
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LinearBetaVAE(input_dim, latent_dim, target_dim, eta_dec_sq, eta_prior_sq, beta).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 50
for epoch in range(epochs):
    total_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        out = model(batch_x, batch_y)
        out['loss'].backward()
        optimizer.step()
        total_loss += out['loss'].item() * batch_x.size(0)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataset):.4f}")


num_samples = 5
z = torch.randn(num_samples, latent_dim).to(device)
generated = model.decoder(z)
print("Generated samples:\n", generated.detach().cpu().numpy())


with torch.no_grad():
    out = model(X_tensor.to(device), X_tensor.to(device))
    mse = torch.mean(torch.square(out['y_pred'] - X_tensor.to(device))).item()
    print(f"Reconstruction MSE: {mse:.4f}")

    gen_mean = generated.mean(0).cpu().numpy()
    gen_std = generated.std(0).cpu().numpy()
    print("Generated mean:", gen_mean)
    print("Generated std:", gen_std)
