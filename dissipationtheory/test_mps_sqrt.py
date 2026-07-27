# John A. Marohn (jam99@cornell.edu)
# 2026-07-26
#
# Google Gemini prompt: I am running macOS, running Pyton 3.10.2, with a "metal" GPU. Write me a Python unit test to check sqrt for complex numbers. The init test should compute on the CPU, in numpy, with double precision and compare against GPU, pytorch. 
#
# Google Gemini result: Here is a complete Python unit test using the standard unittest framework. This test generates complex inputs across all four quadrants (including negative real numbers and negative imaginary components), tests non-contiguous (strided) memory layouts, and compares the PyTorch MPS results against NumPy's high-precision CPU calculations (np.complex128). Key Features of this Unit Test:
# (a) Non-Contiguous Layout Verification: test_non_contiguous_memory_strides creates sliced strided tensors to ensure the MPS backend doesn't read out-of-bounds GPU memory.
# (b) Plug-and-Play Custom Function Testing: You can test any custom fallback function against the same suite by passing custom_fn=your_function into assert_sqrt_matches_numpy.
#
# 
# $ python test_mps_sqrt.py
# 
import unittest
import numpy as np
import torch

class TestComplexSqrtMPS(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Ensure MPS device is available before running GPU tests."""
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("Apple MPS (Metal) GPU is not available on this machine.")
        cls.device = torch.device("mps")

    def assert_sqrt_matches_numpy(self, z_np: np.ndarray, rtol: float = 1e-5, atol: float = 1e-5, custom_fn=None):
        """
        Helper method to compute NumPy CPU reference and compare against PyTorch MPS.
        """
        # 1. Compute high-precision reference on CPU in NumPy (complex128)
        z_np_128 = z_np.astype(np.complex128)
        expected_cpu = np.sqrt(z_np_128)

        # 2. Convert to PyTorch tensor and move to MPS GPU (complex64 / complex128)
        # Note: PyTorch MPS default complex type is complex64
        z_torch = torch.from_numpy(z_np).to(device=self.device)

        # 3. Compute sqrt on GPU (uses custom_fn if passed, else native torch.sqrt)
        if custom_fn is not None:
            actual_mps = custom_fn(z_torch)
        else:
            actual_mps = torch.sqrt(z_torch)

        # 4. Transfer back to CPU as NumPy array for comparison
        actual_cpu = actual_mps.cpu().numpy()

        # 5. Assert closeness
        np.testing.assert_allclose(
            actual_cpu, 
            expected_cpu, 
            rtol=rtol, 
            atol=atol,
            err_msg="PyTorch MPS complex sqrt output does not match NumPy CPU reference!"
        )

    def test_quadrants_and_axes(self):
        """Tests standard complex numbers in all 4 quadrants and along axes."""
        z_np = np.array([
            1.0 + 1.0j,    # Quadrant 1
            -1.0 + 1.0j,   # Quadrant 2
            -1.0 - 1.0j,   # Quadrant 3
            1.0 - 1.0j,    # Quadrant 4
            0.0 + 0.0j,    # Origin
            4.0 + 0.0j,    # Pure positive real
            -4.0 + 0.0j,   # Pure negative real (Branch cut)
            0.0 + 4.0j,    # Pure positive imaginary
            0.0 - 4.0j,    # Pure negative imaginary
        ], dtype=np.complex64)
        
        self.assert_sqrt_matches_numpy(z_np)

    def test_negative_imaginary_inputs(self):
        """Tests inputs specifically with negative imaginary components (b < 0)."""
        rng = np.random.default_rng(42)
        real = rng.uniform(-10.0, 10.0, size=(50,))
        imag = rng.uniform(-10.0, -0.001, size=(50,))  # Strictly negative imaginary
        
        z_np = (real + 1j * imag).astype(np.complex64)
        self.assert_sqrt_matches_numpy(z_np)

    def test_non_contiguous_memory_strides(self):
        """
        Tests strided/non-contiguous tensors on MPS. 
        Catches memory layout bugs caused by slicing or transposing.
        """
        rng = np.random.default_rng(42)
        matrix = rng.uniform(-5.0, 5.0, size=(10, 10)) + 1j * rng.uniform(-5.0, 5.0, size=(10, 10))
        matrix = matrix.astype(np.complex64)

        # Create non-contiguous view via strided slicing
        sliced_np = matrix[::2, ::2]
        
        # Verify that slicing created a non-contiguous tensor in PyTorch
        z_torch_strided = torch.from_numpy(matrix).to(self.device)[::2, ::2]
        self.assertFalse(z_torch_strided.is_contiguous(), "Tensor must be non-contiguous for this test.")

        # Compute CPU reference
        expected_cpu = np.sqrt(sliced_np.astype(np.complex128))
        
        # Compute on MPS GPU
        actual_mps = torch.sqrt(z_torch_strided).cpu().numpy()

        np.testing.assert_allclose(actual_mps, expected_cpu, rtol=1e-5, atol=1e-5)

    def test_branch_cut_float_precision_noise(self):
        """
        Tests behavior along the negative real axis with tiny floating-point noise.
        """
        z_np = np.array([
            -4.0 + 1e-7j,   # Microscopic positive imaginary
            -4.0 - 1e-7j,   # Microscopic negative imaginary
            -100.0 + 0.0j   # Exact zero imaginary
        ], dtype=np.complex64)

        self.assert_sqrt_matches_numpy(z_np)


if __name__ == "__main__":
    unittest.main()