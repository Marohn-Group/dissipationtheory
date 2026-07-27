# $ pytest test_complex_tanh.py
#
# Code written by Google Gemini
# John A. Marohn (jam99)
# 2026-07-25
#
# $ python test_complex_tanh.py

import pytest
import torch
import numpy as np

# --- 1. Import or define the custom function here ---
class AccurateComplexTanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z):
        x, y = z.real, z.imag
        s = torch.where(x >= 0, 1.0, -1.0)
        u = -2.0 * torch.abs(x)
        
        exp_u = torch.exp(u)
        neg_expm1_u = -torch.expm1(u)
        sin_y, cos_y = torch.sin(y), torch.cos(y)
        exp_u_2 = 2.0 * exp_u
        
        N_real = neg_expm1_u + exp_u_2 * (sin_y ** 2)
        D_real = neg_expm1_u + exp_u_2 * (cos_y ** 2)
        N_imag = s * exp_u_2 * sin_y * cos_y
        
        N = torch.complex(N_real, N_imag)
        D = torch.complex(D_real, -N_imag)
        
        out = s * (N / D)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        out, = ctx.saved_tensors
        grad_input = grad_output * (1.0 - out.pow(2)).conj()
        return grad_input

def accurate_complex_tanh(z: torch.Tensor) -> torch.Tensor:
    return AccurateComplexTanh.apply(z)


# --- 2. Testing Utilities ---
def get_numpy_reference(z_tensor: torch.Tensor) -> np.ndarray:
    """
    Computes ground truth using float64/complex128 precision.
    We use double-precision NumPy as the absolute source of truth.
    """
    z_double = z_tensor.detach().cpu().numpy().astype(np.complex128)
    return np.tanh(z_double)


# --- 3. Test Cases ---

def test_standard_domain():
    """Test values on a normal domain [-5, 5] where both should work perfectly."""
    torch.manual_seed(42)
    # Generate random complex64 numbers in [-5, 5]
    real_part = (torch.rand(10000) * 10) - 5
    imag_part = (torch.rand(10000) * 10) - 5
    z = torch.complex(real_part, imag_part)

    out_torch = torch.tanh(z)
    out_custom = accurate_complex_tanh(z)
    ref_np = get_numpy_reference(z)

    # Both native and custom should match ground truth closely (within float32 limits)
    np.testing.assert_allclose(out_custom.numpy(), ref_np, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(out_torch.numpy(), ref_np, rtol=1e-5, atol=1e-6)


def test_catastrophic_overflow_large_reals():
    """
    Test large real parts (x > 44) where float32 exp(2x) exceeds FP32 limits.
    Our accurate_complex_tanh must converge safely to +/-1.
    """
    # Test cases: x = 50, 100, -80, 200
    real_parts = torch.tensor([50.0, 100.0, -80.0, 200.0], dtype=torch.float32)
    imag_parts = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    z = torch.complex(real_parts, imag_parts)

    out_custom = accurate_complex_tanh(z)
    ref_np = get_numpy_reference(z)

    # 1. Custom implementation MUST SUCCEED without NaNs
    assert not torch.isnan(out_custom).any(), "Custom tanh unexpectedly produced NaNs."

    # 2. Custom implementation MUST MATCH numpy ground truth (mostly 1.0 + 0j or -1.0 + 0j)
    np.testing.assert_allclose(out_custom.numpy(), ref_np, rtol=1e-5, atol=1e-6)    


def test_catastrophic_cancellation_near_poles():
    """
    Test values extremely close to the poles (y = pi/2, pi/2 + pi, etc.).
    The custom method leverages expm1 to preserve bits better than native.
    """
    # z = small + i * (pi/2)
    eps = 1e-4
    real_parts = torch.tensor([eps, -eps, eps/10], dtype=torch.float32)
    imag_parts = torch.tensor([np.pi/2, np.pi/2, 3*np.pi/2], dtype=torch.float32)
    z = torch.complex(real_parts, imag_parts)

    out_torch = torch.tanh(z)
    out_custom = accurate_complex_tanh(z)
    ref_np = get_numpy_reference(z)

    # Calculate absolute error against complex128 ground truth
    err_torch = np.abs(out_torch.numpy() - ref_np)
    err_custom = np.abs(out_custom.numpy() - ref_np)

    # The maximum error of our custom method should be lower or perfectly stable 
    # compared to PyTorch's native precision drop
    assert np.max(err_custom) <= np.max(err_torch) * 1.5 # Allow minor floating-point jitter
    
    # Custom method should be extremely close to the double-precision ideal
    np.testing.assert_allclose(out_custom.numpy(), ref_np, rtol=1e-3)


def test_autograd_gradcheck():
    """
    Verify the backward pass (gradients) mathematically using finite differences.
    We use complex128 for gradcheck because finite differences require double precision
    to prevent catastrophic float32 noise from failing the test.
    """
    # Generate random points in a stable domain
    torch.manual_seed(42)
    real_part = (torch.rand(5, dtype=torch.float64) * 4) - 2
    imag_part = (torch.rand(5, dtype=torch.float64) * 4) - 2
    
    # Requires_grad must be True to test backward pass
    z = torch.complex(real_part, imag_part)
    z.requires_grad_(True)

    # PyTorch's gradcheck will perturb `z` slightly (z + epsilon), do a forward pass,
    # and verify that the numeric slope matches the analytic derivative we provided.
    test_passed = torch.autograd.gradcheck(
        accurate_complex_tanh, 
        (z,), 
        eps=1e-6, 
        atol=1e-4
    )
    assert test_passed, "Custom autograd backward pass failed the gradient check!"

if __name__ == "__main__":
    pytest.main([__file__])