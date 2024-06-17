from .bayesPrepClassification import *
import torch.nn.functional as F
import random

# Ref: https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
#     print(f"Random seed set as {seed}")

class BayesConv2d(BayesianModule):

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 kernel_size: int,
                 stride: Optional[int] = 1,
                 padding: Optional[int] = 0,
                 dilation: Optional[int] = 1,
                 prior_pi: Optional[float] = 0.5,
                 prior_sigma1: Optional[float] = 1.0,
                 prior_sigma2: Optional[float] = 0.0025,
                 setseed: int = 2) -> None:


        super().__init__()

        set_seed(setseed)

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = 1



        w_mu = torch.empty(out_features, in_features, *self.kernel_size).uniform_(-.2, .2)
        w_rho = torch.empty(out_features, in_features, *self.kernel_size).uniform_(-5.0, -4.0)

        bias_mu = torch.empty(out_features).uniform_(-.2, .2)
        bias_rho = torch.empty(out_features).uniform_(-5.0, -4.0)

        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)

        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

        self.kl_divergence = 0.0



    def forward(self, x: Tensor) -> Tensor:
        """Calculates the forward pass through the linear layer.

        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear layer.

        Returns
        -------
        Tensor
            Output from the Bayesian Linear layer.
        """

        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()

        w_log_prior = self.w_prior.log_prior(w)
        b_log_prior = self.bias_prior.log_prior(b)

        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()

        total_log_prior = w_log_prior + b_log_prior
        total_log_posterior = w_log_posterior + b_log_posterior

        self.kl_divergence = self.kld(total_log_prior, total_log_posterior)

        return F.conv2d(x, w, b, self.stride, self.padding, self.dilation, self.groups)

    def kld(self, log_prior: Tensor, log_posterior: Tensor) -> Tensor:
        """Calculates the KL Divergence.

        Uses the weight sampled from the posterior distribution to
        calculate the KL Divergence between the prior and posterior.

        Parameters
        ----------
        log_prior : Tensor
            Log likelihood drawn from the prior.
        log_posterior : Tensor
            Log likelihood drawn from the approximate posterior.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        return log_posterior - log_prior



class BayesLinear(BayesianModule):
    """Bayesian Linear Layer.

    Implementation of a Bayesian Linear Layer as described in the
    'Weight Uncertainty in Neural Networks' paper.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 setseed: int = 2,
                 prior_pi: Optional[float] = 0.5,
                 prior_sigma1: Optional[float] = 1.0,
                 prior_sigma2: Optional[float] = 0.0025
                 ) -> None:
        """Bayesian Linear Layer.

        Parameters
        ----------
        in_features : int
            Number of features to feed in to the layer.
        out_features : out
            Number of features produced by the layer.
        prior_pi : float
            Pi weight to be used for the ScaleMixture prior.
        prior_sigma1 : float
            Sigma for the first normal distribution in the prior.
        prior_sigma2 : float
            Sigma for the second normal distribution in the prior.
        """

        super().__init__()

        set_seed(setseed)

        w_mu = torch.randn(out_features, in_features).uniform_(-.01, 0.01)
        w_rho = torch.randn(out_features, in_features).uniform_(-.1, 0.1)

        bias_mu = torch.empty(out_features).uniform_(-.01, 0.01)
        bias_rho = torch.empty(out_features).uniform_(-.1, 0.1)

        self.sth = w_rho
        self.w_posterior = GaussianVariational(w_mu, w_rho)
        self.bias_posterior = GaussianVariational(bias_mu, bias_rho)

        self.w_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)
        self.bias_prior = ScaleMixture(prior_pi, prior_sigma1, prior_sigma2)

        self.kl_divergence = 0.0

    def forward(self, x: Tensor) -> Tensor:
        """Calculates the forward pass through the linear layer.

        Parameters
        ----------
        x : Tensor
            Inputs to the Bayesian Linear layer.

        Returns
        -------
        Tensor
            Output from the Bayesian Linear layer.
        """

        if torch.isnan(self.sth).any() == True:
            print("check nan**:", torch.isnan(self.sth).any())
            print("w**: ", self.sth)

        w = self.w_posterior.sample()
        b = self.bias_posterior.sample()

        w_log_prior = self.w_prior.log_prior(w)
        b_log_prior = self.bias_prior.log_prior(b)

        w_log_posterior = self.w_posterior.log_posterior()
        b_log_posterior = self.bias_posterior.log_posterior()

        total_log_prior = w_log_prior + b_log_prior
        total_log_posterior = w_log_posterior + b_log_posterior

        self.kl_divergence = self.kld(total_log_prior, total_log_posterior)

        return F.linear(x, w, b)

    def kld(self, log_prior: Tensor, log_posterior: Tensor) -> Tensor:
        """Calculates the KL Divergence.

        Uses the weight sampled from the posterior distribution to
        calculate the KL Divergence between the prior and posterior.

        Parameters
        ----------
        log_prior : Tensor
            Log likelihood drawn from the prior.
        log_posterior : Tensor
            Log likelihood drawn from the approximate posterior.

        Returns
        -------
        Tensor
            Calculated KL Divergence.
        """

        return log_posterior - log_prior


@variational_approximator
class BayesianNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, number_layers, setseed=2):
        super().__init__()

        # Set seed
        self.setseed = setseed

        # Fully connected layer
        self.fc1 = BayesLinear(input_dim, 120, setseed=setseed)
        self.fc2 = BayesLinear(120, 84, setseed=setseed)
        self.fc3 = BayesLinear(84, output_dim, setseed=setseed)

        # Number of layers
        self.number_layers = number_layers

    def forward(self, x):

        x = x.view(-1, self.number_layers*1*1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

