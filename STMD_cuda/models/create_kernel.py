import torch
from scipy.special import gamma

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_gamma_kernel(order=100, tau=25, wide=None):
    """
        Generates a discretized Gamma vector as a PyTorch tensor.

        Parameters:
        - order: The order of the Gamma function.
        - tau: The time constant of the Gamma function.
        - wide: The length of the vector.

        Returns:
        - gammaKernel: The generated Gamma vector as a tensor.
        """
    if wide is None:
        wide = int(torch.ceil(torch.tensor(3 * tau)))

    if wide <= 1:
        wide = 2

    timeList = torch.arange(wide, dtype=torch.float32)
    gammaKernel = (
        (order * timeList / tau) ** order *
        torch.exp(-order * timeList / tau) /
        (gamma(order) * tau)
    )

    gammaKernel /= torch.sum(gammaKernel)
    gammaKernel[gammaKernel < 1e-4] = 0
    non_zero_sum = torch.sum(gammaKernel)
    if non_zero_sum > 0:
        gammaKernel /= non_zero_sum
    return gammaKernel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)


def create_fracdiff_kernel(alpha=0.8, wide=3):
    """
        Generates a fractional difference kernel.

        Parameters:
        - alpha: The fractional difference parameter.
        - wide: The width of the kernel.

        Returns:
        - frackernel: The fractional difference kernel.
    """
    # Ensure the width is at least 2
    if wide < 2:
        wide = 2

    # Initialize the kernel
    frackernel = torch.zeros(wide)
    # Generate the kernel based on alpha
    if alpha == 1:
        frackernel[0] = 1
    elif 0 < alpha < 1:
        t_list = torch.arange(wide)
        frackernel = torch.exp(-alpha * t_list / (1 - alpha)) / (1 - alpha)

        # 归一化核
        sum_kernel = torch.sum(frackernel)  # 1/M(α)
        frackernel = frackernel / sum_kernel
        frackernel[frackernel < 1e-4] = 0
        frackernel = frackernel / torch.sum(frackernel)
    else:
        raise ValueError("Alpha 必须在区间 (0,1] 之间。")

    return frackernel.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)


def create_inhi_kernel_W2(kernelSize=15,
                          sigma1=1.5,
                          sigma2=3,
                          e=1.,
                          rho=0,
                          A=1,
                          B=3):
    if kernelSize % 2 == 0:
        kernelSize += 1

    cenX = kernelSize // 2
    cenY = kernelSize // 2

    shiftX, shiftY = torch.meshgrid(torch.arange(1, kernelSize + 1) - cenX,
                                    torch.arange(kernelSize, 0, -1) - cenY)

    gauss1 = (1 / (2 * torch.pi * sigma1 ** 2)) * torch.exp(
        -(shiftX ** 2 + shiftY ** 2) / (2 * sigma1 ** 2))
    gauss2 = (1 / (2 * torch.pi * sigma2 ** 2)) * torch.exp(
        -(shiftX ** 2 + shiftY ** 2) / (2 * sigma2 ** 2))

    dogFilter = gauss1 - e * gauss2 - rho

    positiveComponent = torch.clamp(dogFilter, min=0)
    negativeComponent = torch.clamp(dogFilter, max=0)

    inhibitionKernelW2 = A * positiveComponent + B * negativeComponent

    return inhibitionKernelW2.unsqueeze(0).unsqueeze(0).to(device)

if __name__ == '__main__':
    a = create_gamma_kernel(2,3,3)
    print(a)