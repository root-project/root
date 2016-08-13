// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////
// Definition of the TCuda architecture, which provides an   //
// implementation of the low-level functionality for neural  //
// networks for the CUDA computing architectures.            //
///////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDA
#define TMVA_DNN_ARCHITECTURES_CUDA

#include <utility>

#include "cuda.h"

#include "Cuda/Types.h"
#include "Cuda/Kernels.h"
#include "Cuda/Buffers.h"
#include "Cuda/DataLoader.h"
#include "Cuda/CudaMatrix.h"
#include "TMVA/DNN/DataLoader.h"

namespace TMVA
{
namespace DNN
{

/** The TCuda architecture class.
 *
 * Low-level interface class for CUDA computing architecture. Contains as
 * public types the declaration of the scalar, matrix and data loader types
 * for this architecture as well as the remaining functions in the low-level
 * interface in the form of static members.
 */
class TCuda
{

public:

    using Scalar_t       = CudaDouble_t;
    using Matrix_t       = TCudaMatrix;
    using DeviceBuffer_t = TCudaDeviceBuffer;
    using HostBuffer_t   = TCudaHostBuffer;
    template <typename Data_t>
    using DataLoader_t   = TCudaDataLoader<Data_t>;

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Matrix-multiply \p input with the transpose of \pweights and
    *  write the results into \p output. */
   static void MultiplyTranspose(TCudaMatrix &output,
                                 const TCudaMatrix &input,
                                 const TCudaMatrix &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TCudaMatrix &output,
                          const TCudaMatrix &biases);
   ///@}

   /** @name Backward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */
   ///@{
   /** Perform the complete backward propagation step. If the provided
    *  \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void Backward(TCudaMatrix & activationGradientsBackward,
                        TCudaMatrix & weightGradients,
                        TCudaMatrix & biasGradients,
                        TCudaMatrix & df,
                        const TCudaMatrix & activationGradients,
                        const TCudaMatrix & weights,
                        const TCudaMatrix & activationBackward);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TCudaMatrix & A,
                        const TCudaMatrix & B,
                        Scalar_t beta = 1.0);

   static void Copy(TCudaMatrix & B,
                    const TCudaMatrix & A);
   ///@}

   //____________________________________________________________________________
   //
   // Activation Functions
   //____________________________________________________________________________

   /** @name Activation Functions
    * For each activation function, the low-level interface contains two routines.
    * One that applies the acitvation function to a matrix and one that evaluate
    * the derivatives of the activation function at the elements of a given matrix
    * and writes the results into the result matrix.
    */
   ///@{
   static void Identity(TCudaMatrix & B);
   static void IdentityDerivative(TCudaMatrix & B,
                                  const TCudaMatrix & A);

   static void Relu(TCudaMatrix & B);
   static void ReluDerivative(TCudaMatrix & B,
                              const TCudaMatrix & A);

   static void Sigmoid(TCudaMatrix & B);
   static void SigmoidDerivative(TCudaMatrix & B,
                                 const TCudaMatrix & A);

   static void Tanh(TCudaMatrix & B);
   static void TanhDerivative(TCudaMatrix & B,
                              const TCudaMatrix & A);

   static void SymmetricRelu(TCudaMatrix & B);
   static void SymmetricReluDerivative(TCudaMatrix & B,
                                       const TCudaMatrix & A);

   static void SoftSign(TCudaMatrix & B);
   static void SoftSignDerivative(TCudaMatrix & B,
                                  const TCudaMatrix & A);

   static void Gauss(TCudaMatrix & B);
   static void GaussDerivative(TCudaMatrix & B,
                               const TCudaMatrix & A);
   ///@}

   //____________________________________________________________________________
   //
   // Loss Functions
   //____________________________________________________________________________

   /** @name Loss Functions
    * Loss functions compute a scalar value given the \p output of the network
    * for a given training input and the expected network prediction \p Y that
    * quantifies the quality of the prediction. For each function also a routing
    * that computes the gradients (suffixed by Gradients) must be provided for
    * the starting of the backpropagation algorithm.
    */
   ///@{

   static CudaDouble_t MeanSquaredError(const TCudaMatrix &Y,
                                        const TCudaMatrix &output);
   static void MeanSquaredErrorGradients(TCudaMatrix & dY,
                                         const TCudaMatrix &Y,
                                         const TCudaMatrix &output);

    /** Sigmoid transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static CudaDouble_t CrossEntropy(const TCudaMatrix &Y,
                              const TCudaMatrix &output);

   static void CrossEntropyGradients(TCudaMatrix & dY,
                                     const TCudaMatrix & Y,
                                     const TCudaMatrix & output);
   ///@}

   //____________________________________________________________________________
   //
   // Output Functions
   //____________________________________________________________________________

   /** @name Output Functions
    * Output functions transform the activations \p output of the
    * output layer in the network to a valid prediction \p YHat for
    * the desired usage of the network, e.g.  the identity function
    * for regression or the sigmoid transformation for two-class
    * classification.
    */
   ///@{
   static void Sigmoid(TCudaMatrix &YHat,
                        const TCudaMatrix & );
   ///@}

   //____________________________________________________________________________
   //
   // Regularization
   //____________________________________________________________________________

   /** @name Regularization
    * For each regularization type two functions are required, one named
    * <tt><Type>Regularization</tt> that evaluates the corresponding
    * regularization functional for a given weight matrix and the
    * <tt>Add<Type>RegularizationGradients</tt>, that adds the regularization
    * component in the gradients to the provided matrix.
    */
   ///@{

   static CudaDouble_t L1Regularization(const TCudaMatrix & W);
   static void AddL1RegularizationGradients(TCudaMatrix & A,
                                            const TCudaMatrix & W,
                                            CudaDouble_t weightDecay);

   static CudaDouble_t L2Regularization(const TCudaMatrix & W);
   static void AddL2RegularizationGradients(TCudaMatrix & A,
                                            const TCudaMatrix & W,
                                            CudaDouble_t weightDecay);
   ///@}

   //____________________________________________________________________________
   //
   // Initialization
   //____________________________________________________________________________

   /** @name Initialization
    * For each initialization method, one function in the low-level interface
    * is provided. The naming scheme is <p>Initialize<Type></p> for a given
    * initialization method Type.
    */
   ///@{

   static void InitializeGauss(TCudaMatrix & A);
   static void InitializeUniform(TCudaMatrix & A);
   static void InitializeIdentity(TCudaMatrix & A);
   static void InitializeZero(TCudaMatrix & A);

   ///@}

   //____________________________________________________________________________
   //
   // Dropout
   //____________________________________________________________________________

   /** @name Dropout
    */
   ///@{

   /** Apply dropout with activation probability \p p to the given
    *  matrix \p A and scale the result by reciprocal of \p p. */
   static void Dropout(TCudaMatrix & A, CudaDouble_t p);

   ///@}

   //____________________________________________________________________________
   //
   // Additional Arithmetic Functions
   //____________________________________________________________________________

   /** @name Additional Arithmetic Functions
    *
    * Additional arithmetic on CUDA matrices  used to implement the low-level
    * interface.
    */
   ///@{

   /** Standard multiplication of two matrices \p A and \p B with the result being
    *  written into C.
    */
   static void Multiply(TCudaMatrix &C,
                        const TCudaMatrix &A,
                        const TCudaMatrix &B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(TCudaMatrix &output,
                                 const TCudaMatrix &input,
                                 const TCudaMatrix &Weights);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(TCudaMatrix &A,
                        const TCudaMatrix &B);

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(TCudaMatrix &B, const TCudaMatrix &A);

   /** Compute the sum of all elements in \p A */
   static CudaDouble_t Sum(const TCudaMatrix &A);
};

} // namespace DNN
} // namespace TMVA

#endif
