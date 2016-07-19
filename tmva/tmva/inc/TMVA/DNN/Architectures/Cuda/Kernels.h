// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Declaration of the device kernels for the CUDA implementation of //
// the low-level interface.                                         //
//////////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "cuda.h"
#include "curand_kernel.h"

namespace TMVA {
namespace DNN  {
namespace Cuda {


/** @name Device Kernels
 * Utility device kernels that can only be called from the device.
 */
///@{
/** Atomic addition of doubles, which is at the moment natively supported only
 for floats. */
//____________________________________________________________________________
__device__ CudaDouble_t AtomicAdd(CudaDouble_t* address, CudaDouble_t val);

/** Sum elements in columns of the 2D shared data array and accumulated the
 *  results in \p result.
 *
 *  \param result Pointer to the result where the reduction results of each block
 *  will be accumulated using atomicAdd. The results of the jth column in this
 *  2D block will be accumulated in the jth element.
 *  \param sdata 2D shared data array with the same dimensions as the current
 *  compute block. Contains the elements that will be row-wise accumulated.
 */
//____________________________________________________________________________
__device__ void ReduceSumVertical(CudaDouble_t *result, CudaDouble_t * sdata);

/** Sum up all elements in the current compute block and accumulate the results
 *  into \p result using atomicAdd */
//____________________________________________________________________________
__device__ void ReduceSum(CudaDouble_t *result, CudaDouble_t * sdata);
///@}

/** @name Forward and Backward Propagation
 */
///@{
/** Add the \p n -element vector \p theta row-wise to the \p m x \p n matrix
 * in W */
//____________________________________________________________________________
__global__ void AddRowWise(CudaDouble_t * W, const CudaDouble_t * theta,
                             int m, int n);
/** Compute the Hadamard product of the \p m x \p n matrix B and A and write
 * results into B. */
//____________________________________________________________________________
__global__ void Hadamard(CudaDouble_t * B,
                         const CudaDouble_t * A,
                         int m, int n);
///@}

/** @name Activation Functions
 */
///@{
//____________________________________________________________________________
__global__ void IdentityDerivative(CudaDouble_t * A,
                                   int m, int n);
//____________________________________________________________________________
__global__ void Relu(CudaDouble_t * A,
                     int m, int n);

//____________________________________________________________________________
__global__ void ReluDerivative(CudaDouble_t * B,
                               const CudaDouble_t * A, int m, int n);

//____________________________________________________________________________
__global__ void Sigmoid(CudaDouble_t * A,
                        int m, int n);
//____________________________________________________________________________
__global__ void SigmoidDerivative(CudaDouble_t * B,
                                  const CudaDouble_t * A,
                                   int m, int n);

//____________________________________________________________________________
__global__ void Tanh(CudaDouble_t * A,
                     int m, int n);
//____________________________________________________________________________
__global__ void TanhDerivative(CudaDouble_t * B,
                               const CudaDouble_t * A,
                               int m, int n);

//____________________________________________________________________________
__global__ void SymmetricRelu(CudaDouble_t * A,
                              int m, int n);
//____________________________________________________________________________
__global__ void SymmetricReluDerivative(CudaDouble_t * B,
                                        const CudaDouble_t * A,
                                        int m, int n);

//____________________________________________________________________________
__global__ void SoftSign(CudaDouble_t * A,
                         int m, int n);
//____________________________________________________________________________
__global__ void SoftSignDerivative(CudaDouble_t * B,
                                   const CudaDouble_t * A,
                                   int m, int n);

//____________________________________________________________________________
__global__ void Gauss(CudaDouble_t * A,
                      int m, int n);
//____________________________________________________________________________
__global__ void GaussDerivative(CudaDouble_t * B,
                                const CudaDouble_t * A,
                                int m, int n);

//____________________________________________________________________________
///@}

/** @name Loss Functions
 */
///@{

//____________________________________________________________________________
__global__ void MeanSquaredError(CudaDouble_t * result,
                                 const CudaDouble_t * Y,
                                 const CudaDouble_t * output,
                                 int m, int n);

//____________________________________________________________________________
__global__ void MeanSquaredErrorGradients(CudaDouble_t * dY,
                                          const CudaDouble_t * Y,
                                          const CudaDouble_t * output,
                                          int m, int n);
//____________________________________________________________________________
__global__ void CrossEntropy(CudaDouble_t * result,
                             const CudaDouble_t * Y,
                             const CudaDouble_t * output,
                             int m, int n);

//____________________________________________________________________________
__global__ void CrossEntropyGradients(CudaDouble_t * dY,
                                      const CudaDouble_t * Y,
                                      const CudaDouble_t * output,
                                      int m, int n);
///@}

/** @name Regularization
 */
///@{

/** Compute the sum of the absolute values of the elements in the \p m x \p n
 *  matrix \p A and write the result into \p result. This is used to compute
 *  L1 regularization for weights matrices. */
//____________________________________________________________________________
__global__ void AbsoluteSum(CudaDouble_t * result,
                            const CudaDouble_t * A,
                            int m, int n);

/** Compute the squared sum of the \p m x \p n matrix \p A and write the result
 *  into \p result. This is used to compute L2 regularization. */
//____________________________________________________________________________
__global__ void SquaredSum(CudaDouble_t * result,
                           const CudaDouble_t * A,
                           int m, int n);

/** Add the gradients of L1 regularizatoin applied to the \p m x \p n matrix
 * in \p A to the \p m x \p n matrix in B. \p weightDecay is the weight assigned
 * to the L1 regularization term. */
//____________________________________________________________________________
__global__ void AddL1RegularizationGradients(CudaDouble_t * B,
                                             const CudaDouble_t * A,
                                             CudaDouble_t weightDecay,
                                             int m, int n);

/** Add the gradients of L2 regularizatoin applied to the \p m x \p n matrix
 * in \p A to the \p m x \p n matrix in B. \p weightDecay is the weight assigned
 * to the L1 regularization term. */
//____________________________________________________________________________
__global__ void AddL2RegularizationGradients(CudaDouble_t * A,
                                             const CudaDouble_t * B,
                                             CudaDouble_t weightDecay,
                                             int m, int n);

///@}

///@{
/** @name Dropout
 */
//____________________________________________________________________________
__global__ void InitializeCurandStates(unsigned long long seed,
                                       curandState_t *states);

//____________________________________________________________________________
__global__ void Dropout(CudaDouble_t *A,
                        int m, int n,
                        CudaDouble_t dropoutProbability,
                        curandState_t *states);

///@}

///@{
/** @name Output Functions
 */
//____________________________________________________________________________
__global__ void Sigmoid(CudaDouble_t *A,
                        const CudaDouble_t *B,
                        int m, int n);

///@}

///@{
/** @name Miscellaneous
 */
///@{

//____________________________________________________________________________
__global__ void ReduceMatrix(CudaDouble_t *result,
                             const CudaDouble_t *A,
                             int m, int n);

//____________________________________________________________________________
__global__ void SumColumns(CudaDouble_t *B,
                           const CudaDouble_t *A,
                           int m, int n);
///@}

} // namespace CudaKernels
} // namespace DNN
} // namespace TMVA
