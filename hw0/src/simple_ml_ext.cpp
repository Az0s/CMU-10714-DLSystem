#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

     /// BEGIN YOUR CODE
    for (size_t i = 0; i < m; i += batch) {
        size_t batch_size = std::min(batch, m - i);
        // Allocate memory for logits and gradients
        float* Z = new float[batch_size* k];
        float* expZ = new float[batch_size* k];
        float* softmax_probs = new float[batch_size * k];
        float* I_y = new float[batch_size * k];
        float* gradients = new float[n * k];
        // Initialize gradients to zero
        for (size_t j = 0; j < n * k; ++j) {
            gradients[j] = 0.0f;
        }
        // forward 
        // X_batch(B, n) @ theta (n, k)
        for (size_t r = 0; r < batch_size; ++r) {
            for (size_t c = 0; c < k; ++c) {
                Z[r*k + c] = 0.0f;
                for (size_t d = 0; d < n; ++d) {
                    Z[r * k + c] += X[(i + r) * n + d] * theta[d * k + c];
                }
            }
        }
        // np.exp(Z) / np.exp(Z).sum(axis=1, keepdims=True) # B, k
        for (size_t r = 0; r < batch_size; ++r) {
            float sum_expZ = 0.0f;
            for (size_t c = 0; c < k; ++c) {
                expZ[r * k + c] = std::exp(Z[r * k + c]);
                sum_expZ += expZ[r * k + c];
            }
            for (size_t c = 0; c < k; ++c) {
                softmax_probs[r * k + c] = expZ[r * k + c] / sum_expZ;
            }
        }
        // I_y 
        for (size_t r = 0; r < batch_size; ++r) {
            for (size_t c = 0; c < k; ++c) {
                I_y[r * k + c] = (c == y[i + r]) ? 1.0f : 0.0f;
            }
        }
        // TODO: optimization 
        float* X_batch_T = new float[ n * batch_size];
        // Transpose X
        for (size_t r = 0; r < batch_size; ++r) {
            for (size_t c = 0; c < n; ++c) {
                X_batch_T[c * batch_size + r] = X[(i + r) * n + c];
            }
        }
        //  X_batch.T @ (z - I_y) / z.shape[0] # n, k
        // !flawed
        // for (size_t r = 0; r < n; ++r) {
        //     for (size_t c = 0; c < k; ++c) {
                
        //         gradients[r * k + c] += X_batch_T[r * batch_size + c] * (softmax_probs[c] - I_y[c]) / batch_size;
        //     }
        // }
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < k; ++c) {
                gradients[r * k + c] = 0.0f; // Reset for this batch
                for (size_t b = 0; b < batch_size; ++b) {
                    gradients[r * k + c] += X_batch_T[r * batch_size + b] * (softmax_probs[b * k + c] - I_y[b * k + c]);
                }
                gradients[r * k + c] /= batch_size;
            }
        }
        // Update theta
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < k; ++c) {
                theta[r * k + c] -= lr * gradients[r * k + c];
            }
        }
        // Free memory
        delete[] Z;
        delete[] expZ;
        delete[] softmax_probs;
        delete[] I_y;
        delete[] X_batch_T;
    }
    return;
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
