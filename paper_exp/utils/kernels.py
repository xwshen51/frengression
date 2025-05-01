import jax
import jax.numpy as jnp


def l2norm(X, Y):
    """Compute \|X - Y\|_2^2 of tensors X, Y
    Args:
        X: Tensors of shape (..., n, dim)
        Y: Tensors of shape (..., m, dim)
    """
    XY = jnp.matmul(X, jnp.moveaxis(Y, (-1, -2), (-2, -1))) # n x m
    XX = jnp.matmul(X, jnp.moveaxis(X, (-1, -2), (-2, -1)))
    XX = jnp.expand_dims(jnp.diagonal(XX, axis1=-2, axis2=-1), axis=-1) # n x 1
    YY = jnp.matmul(Y, jnp.moveaxis(Y, (-1, -2), (-2, -1)))
    YY = jnp.expand_dims(jnp.diagonal(YY, axis1=-2, axis2=-1), axis=-2) # 1 x m

    dnorm2 = -2 * XY + XX + YY
    return dnorm2

def median_heuristic(dnorm2):
    """Compute median heuristic med(\|X_i - X_j\|_2^2, 1 \leq i < j \leq n).
    
    :param dnorm2: (n, n) tensor of \|X - Y\|_2^2
    """
    ind_array = jnp.triu(jnp.ones_like(dnorm2), k=1) == 1
    med_heuristic = jnp.percentile(dnorm2[ind_array], 50.0)
    return med_heuristic

class Kernel(object):
    """A kernel class need to have the following methods:
        __call__: k(x, y)
        grad_first: \nabla_x k(x, y)
        grad_second: \nabla_y k(x, y)
        gradgrad: \nabla_x \nabla_y k(x, y)
    """

    def __call__(self, X, Y):
        """Compute k(x, y)

        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)
        
        :return jnp.array of shape (..., n, m)
        """
        raise NotImplementedError
    
    def grad_first(self, X, Y):
        """Compute \nabla_x k(x, y) wrt the first argument.
        
        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)

        :return jnp.array of shape (..., n, m, dim)
        """
        raise NotImplementedError

    def grad_second(self, X, Y):
        """Compute \nabla_x k(x, y) wrt the first argument.

        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)
        
        :return jnp.array of shape (..., n, m, dim)
        """
        raise NotImplementedError

    def gradgrad(self, X, Y):
        """Compute \nabla_x^\top \nabla_y k(x, y).

        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)
        
        :return jnp.array of shape (..., n, m)
        """
        raise NotImplementedError

class RBF(Kernel):
    """RBF kernel k(x, y) = exp(-\|x - y\|_2^2 / \sigma^2)
    """

    def __init__(self, sigma_sq=None, med_heuristic=False, scale=1., X=None, Y=None):
        """
        :param sigma_sq: float, squared bandwidth parameter \sigma^2
        :param med_heuristic: bool, whether to use median heuristic for bandwidth. If True,
            X, Y must be provided to compute the median heuristic.
        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)
        """
        super().__init__()
        self.sigma_sq = sigma_sq
        self.med_heuristic = med_heuristic
        self.scale = scale

        if med_heuristic:
            assert X is not None and Y is not None, "Need to provide X, Y for med heuristic"
            self.bandwidth(X, Y)

    def UB(self):
        return self.scale

    def bandwidth(self, X, Y):
        """Compute med heuristic for bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        sigma2 = med_heuristic_sq
        self.sigma_sq = sigma2
    
    def __call__(self, X, Y):
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-12)
        sigma2_inv = jnp.expand_dims(jnp.expand_dims(sigma2_inv, -1), -1)
        K_XY = jnp.exp(- sigma2_inv * dnorm2)

        return self.scale * K_XY

    def grad_first(self, X, Y):
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        K = jnp.expand_dims(jnp.exp(- l2norm(X, Y) * sigma2_inv), -1) # n x m x 1
        # diff_{ijk} = y^i_j - x^i_k
        Yp = jnp.expand_dims(Y, -3) # 1 x m x dim
        Xp = jnp.expand_dims(X, -2) # n x 1 x dim
        diff = Yp - Xp # n x m x dim
        # compute grad_K
        grad_K_XY = - 2 * sigma2_inv * diff * K # n x m x dim
        return self.scale * grad_K_XY

    def gradgrad(self, X, Y):
        # Gram matrix
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        diff_norm_sq = l2norm(X, Y) # n x m
        K = jnp.exp(-l2norm(X, Y) * sigma2_inv) # n x m
        term1 = 2 * sigma2_inv * X.shape[-1]
        term2 = - 4 * sigma2_inv ** 2 * diff_norm_sq # n x m
        gradgrad_tr = (term1 + term2) * K # n x m
        return self.scale * gradgrad_tr

class IMQ(Kernel):
    """IMQ kernel k(x, y) = (1 + \|x - y\|_2^2 / \sigma^2)^\beta
    """

    def __init__(self, sigma_sq=None, beta=-0.5, med_heuristic=False, X=None, Y=None):
        """
        :param sigma_sq: float, squared bandwidth parameter \sigma^2
        :param med_heuristic: bool, whether to use median heuristic for bandwidth. If True,
            X, Y must be provided to compute the median heuristic.
        :param X: jnp.array of shape (..., n, dim)
        :param Y: jnp.array of shape (..., m, dim)
        """
        super().__init__()
        self.sigma_sq = sigma_sq
        self.beta = beta
        self.med_heuristic = med_heuristic

        if med_heuristic:
            assert X is not None and Y is not None, "Need to provide X, Y for med heuristic"
            self.bandwidth(X, Y)
    
    def bandwidth(self, X, Y):
        """Compute med heuristic for bandwidth
        """
        dnorm2 = l2norm(X, Y)
        med_heuristic_sq = median_heuristic(dnorm2)
        self.sigma_sq = med_heuristic_sq
        
    def __call__(self, X, Y):
        dnorm2 = l2norm(X, Y)
        sigma2_inv = 1.0 / (self.sigma_sq + 1e-12)
        sigma2_inv = jnp.expand_dims(jnp.expand_dims(sigma2_inv, -1), -1)
        K_XY = jax.lax.pow(1 + sigma2_inv * dnorm2, self.beta)

        return K_XY

    def grad_first(self, X, Y):
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        K = 1. + jnp.expand_dims(l2norm(X, Y) * sigma2_inv, -1) # n x m x 1
        # diff_{ijk} = y^k_i - x^k_j
        diff = jnp.expand_dims(Y, -3) - jnp.expand_dims(X, -2) # n x m x dim
        # compute grad_K
        grad_K_XY = 2 * sigma2_inv * diff * self.beta * jax.lax.pow(K, self.beta-1) # n x m x dim

        return grad_K_XY   

    def gradgrad(self, X, Y):
        sigma2_inv = 1 / (1e-12 + self.sigma_sq)
        # norm of differences
        diff_norm_sq = l2norm(X, Y) # n x m
        K = 1 + diff_norm_sq * sigma2_inv # n x m
        term1 = - 2 * sigma2_inv * self.beta * X.shape[-1] * K # n x m
        term2 = - self.beta * (self.beta-1) * 4 * sigma2_inv**2 * diff_norm_sq # n x m
        gradgrad_tr = (
            term1 + term2
        ) * jax.lax.pow(K, self.beta-2) # n x m

        return gradgrad_tr

class TiltedKernel(Kernel):
    """Tilted kernel k(x, y) = w(x) k_0(x, y) w(y)
    """

    def __init__(self, kernel, weight_fn):
        """
        :param kernel: Base kernel k_0(x, y)
        :param weight_fn: Weighting function w
        """
        super().__init__()
        self.base_kernel = kernel
        self.weight_fn = weight_fn
        self.med_heuristic = None

    def __call__(self, X, Y):
        K_XY = self.base_kernel(X, Y) # n, m
        W_X = self.weight_fn(X) # n
        W_Y = self.weight_fn(Y) # m
        res = jnp.expand_dims(W_X, -1) * K_XY * jnp.expand_dims(W_Y, -2) # n, m
        return res

    def grad_first(self, X, Y):
        K = self.base_kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.base_kernel.grad_first(X, Y)
        grad_W_X = self.weight_fn.grad(X) # n, d
        W_Y_pd = jnp.expand_dims(jnp.expand_dims(W_Y, -2), -1) # 1, m, 1

        term1 = jnp.expand_dims(grad_W_X, -2) * jnp.expand_dims(K, -1) * W_Y_pd # n, m, d
        term2 = W_X[..., jnp.newaxis, jnp.newaxis] * grad_K_XY * W_Y_pd # n, m, d
        return term1 + term2

    def grad_second(self, X, Y):
        K = self.base_kernel(X, Y)
        W_X = self.weight_fn(X)
        W_Y = self.weight_fn(Y)
        grad_K_XY = self.base_kernel.grad_second(X, Y) # n, m, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        W_X_pd = W_X[..., jnp.newaxis, jnp.newaxis] # n, 1, 1

        term1 = W_X_pd * grad_K_XY * jnp.expand_dims(
            jnp.expand_dims(W_Y, -2), -1
        ) # n, m, d
        term2 = W_X_pd * jnp.expand_dims(K, -1) * jnp.expand_dims(grad_W_Y, -3) # n, m, d
        return term1 + term2

    def gradgrad(self, X, Y):
        K = self.base_kernel(X, Y)
        grad_K_X = self.base_kernel.grad_first(X, Y) # n, m, d
        grad_K_Y = self.base_kernel.grad_second(X, Y) # n, m, d
        gradgrad_K = self.base_kernel.gradgrad(X, Y) # n, m
        
        W_X = self.weight_fn(X)
        W_X_pd = jnp.expand_dims(W_X, -1) # n, 1
        W_Y = self.weight_fn(Y)
        W_Y_pd = jnp.expand_dims(W_Y, -2) # 1, m

        grad_W_X = self.weight_fn.grad(X) # n, d
        grad_W_X_pd = jnp.expand_dims(grad_W_X, -2) # n, 1, d
        grad_W_Y = self.weight_fn.grad(Y) # m, d
        grad_W_Y_pd = jnp.expand_dims(grad_W_Y, -3) # 1, m, d

        term1 = jnp.sum(grad_W_X_pd * grad_K_Y, -1) * W_Y_pd # n, m
        term2 = W_X_pd * gradgrad_K * W_Y_pd # n, m
        term3 = jnp.sum(grad_W_X_pd * grad_W_Y_pd, -1) * K # n, m
        term4 = W_X_pd * jnp.sum(grad_K_X * grad_W_Y_pd, -1) # n, m

        return term1 + term2 + term3 + term4

class SumKernel(Kernel):
    """Sum of kernels k(x, y) = \sum_{l=1}^L k_l(x, y).
    """
    
    def __init__(self, kernels):
        """
        :param kernels: list of Kernel objects
        """
        super().__init__()
        self.kernels = kernels

    def __call__(self, X, Y):
        res = 0.
        for kernel in self.kernels:
            res += kernel(X, Y)

        return res

    def grad_first(self, X, Y):
        return -self.grad_second(X, Y)

    def grad_second(self, X, Y):
        res = 0.
        for kernel in self.kernels:
            res += kernel.grad_second(X, Y)

        return res

    def gradgrad(self, X, Y):
        res = 0.
        for kernel in self.kernels:
            res += kernel.gradgrad(X, Y)

        return res
    
class WeightFunction(object):
    """Weighting function w: R^d \to [0, \infty)
    """

    def __call__(self, X):
        """Compute w(x)

        :param X: jnp.array of shape (..., n, dim)
        """
        raise NotImplementedError

    def grad(self, X):
        """Compute \nabla w(x)

        :param X: jnp.array of shape (..., n, dim)
        """
        raise NotImplementedError

    def set_sup(self, weighted_score_sup, sup, derivative_sup):
        self.weighted_score_sup = weighted_score_sup
        self.sup = sup
        self.derivative_sup = derivative_sup

class PolyWeightFunction(WeightFunction):
    """Polynomial weighting function w(x) = (1 + \|x - loc\|_2^2 / a^2)^(-b)
    """

    def __init__(self, b = 0.5, loc = 0., a = 1., weighted_score_sup: float = None):
        """
        :param b: float, must be positive
        :param loc: jnp.array of shape (dim,)
        :param a: float, must be positive
        """
        self.loc = jnp.array(loc)
        self.b = b
        assert self.b > 0.
        self.a = a

        self.weighted_score_sup = 1 / self.a if weighted_score_sup is None else weighted_score_sup
        self.sup = 1.
        self.derivative_sup = 2. * self.b * self.a

    def __call__(self, X):
        assert jnp.squeeze(self.loc).shape == () or jnp.squeeze(X[0]).shape == jnp.squeeze(self.loc).shape

        score_norm_sq = jnp.sum((X - self.loc)**2, -1) # n
        return jax.lax.pow(1 + score_norm_sq / self.a**2, -self.b) # n

    def grad(self, X):
        score_norm_sq = jnp.sum((X - self.loc)**2, -1)

        res = -2 * self.b * jnp.expand_dims(
            jax.lax.pow(1 + score_norm_sq / self.a**2, -self.b - 1),
            -1,
        ) * (X - self.loc) * self.a**(-2) # n, d
        return res