// @(#)root/tmva/tmva/dnn:$Id$
// Author: Joana Niermann 23/07/19

/*************************************************************************
 * Copyright (C) 2019, Joana Niermann                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Definition of the TCudnn architecture class, which provides   //
// a wrapping of the low-level functionality for neural networks //
// in the cuDNN library.                                         //
///////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CUDNN
#define TMVA_DNN_ARCHITECTURES_CUDNN

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ConvLayer.h"

#include "cudnn.h"
#include "Cuda/CudaBuffers.h"
#include "Cudnn/CudnnTensor.h"
#include "TMVA/DNN/TensorDataLoader.h"
#include <utility>
#include <vector>

class TRandom;

namespace TMVA
{
namespace DNN
{

/** The TCudnn architecture class.
 *
 * Low-level interface class for CUDA computing architectures using the cuDNN
 * library as backend. Contains as public types the declaration of the scalar, 
 * matrix and buffer types for this architecture, as well as the remaining 
 * functions in the low-level interface in the form of static members.
 */
template<typename AFloat = cudnnDataType_t>
class TCudnn
{
private:
   static TRandom * fgRandomGen;
public:

    using Scalar_t       = AFloat;
    using Matrix_t       = TCudnnTensor<AFloat>;
    using DeviceBuffer_t = TCudaDeviceBuffer<AFloat>;
    using HostBuffer_t   = TCudaHostBuffer<AFloat>;

   //____________________________________________________________________________
   //
   // Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation
    * Low-level functions required for the forward propagation of activations
    * through the network.
    */

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
   static void Backward();
   
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TCudnnTensor<AFloat> & A,
                        const TCudnnTensor<AFloat> & B,
                        Scalar_t beta = 1.0);
   /** Copy the elements of matrix A into matrix B. */
   static void Copy(TCudnnTensor<AFloat> & B,
                    const TCudnnTensor<AFloat> & A);

   // copy from another type of matrix
   template<typename AMatrix_t>
   static void CopyDiffArch(TCudnnTensor<Scalar_t> & B, const AMatrix_t & A); 


   /** Above functions extended to vectors */
   static void ScaleAdd(std::vector<TCudnnTensor<Scalar_t>> & A,
                        const std::vector<TCudnnTensor<Scalar_t>> & B,
                        Scalar_t beta = 1.0);

   static void Copy(std::vector<TCudnnTensor<Scalar_t>> & A,
                    const std::vector<TCudnnTensor<Scalar_t>> & B);

   // copy from another architecture
   template<typename AMatrix_t>
   static void CopyDiffArch(std::vector<TCudnnTensor<Scalar_t>> & A,
const std::vector<AMatrix_t> & B);

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
   static void Identity(TCudnnTensor<AFloat> & B);
   static void IdentityDerivative(TCudnnTensor<AFloat> & B,
                                  const TCudnnTensor<AFloat> & A);

   static void Relu(TCudnnTensor<AFloat> & B);
   static void ReluDerivative(TCudnnTensor<AFloat> & B,
                              const TCudnnTensor<AFloat> & A);

   static void Sigmoid(TCudnnTensor<AFloat> & B);
   static void SigmoidDerivative(TCudnnTensor<AFloat> & B,
                                 const TCudnnTensor<AFloat> & A);

   static void Tanh(TCudnnTensor<AFloat> & B);
   static void TanhDerivative(TCudnnTensor<AFloat> & B,
                              const TCudnnTensor<AFloat> & A);

   static void SymmetricRelu(TCudnnTensor<AFloat> & B);
   static void SymmetricReluDerivative(TCudnnTensor<AFloat> & B,
                                       const TCudnnTensor<AFloat> & A);

   static void SoftSign(TCudnnTensor<AFloat> & B);
   static void SoftSignDerivative(TCudnnTensor<AFloat> & B,
                                  const TCudnnTensor<AFloat> & A);

   static void Gauss(TCudnnTensor<AFloat> & B);
   static void GaussDerivative(TCudnnTensor<AFloat> & B,
                               const TCudnnTensor<AFloat> & A);
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

   static AFloat MeanSquaredError(const TCudnnTensor<AFloat> &Y, const TCudnnTensor<AFloat> &output,
                                  const TCudnnTensor<AFloat> &weights);
   static void MeanSquaredErrorGradients(TCudnnTensor<AFloat> &dY, const TCudnnTensor<AFloat> &Y,
                                         const TCudnnTensor<AFloat> &output, const TCudnnTensor<AFloat> &weights);

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static AFloat CrossEntropy(const TCudnnTensor<AFloat> &Y, const TCudnnTensor<AFloat> &output,
                              const TCudnnTensor<AFloat> &weights);

   static void CrossEntropyGradients(TCudnnTensor<AFloat> &dY, const TCudnnTensor<AFloat> &Y,
                                     const TCudnnTensor<AFloat> &output, const TCudnnTensor<AFloat> &weights);

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static AFloat SoftmaxCrossEntropy(const TCudnnTensor<AFloat> &Y, const TCudnnTensor<AFloat> &output,
                                     const TCudnnTensor<AFloat> &weights);
   static void SoftmaxCrossEntropyGradients(TCudnnTensor<AFloat> &dY, const TCudnnTensor<AFloat> &Y,
                                            const TCudnnTensor<AFloat> &output, const TCudnnTensor<AFloat> &weights);
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
   static void Sigmoid(TCudnnTensor<AFloat> &YHat,
                       const TCudnnTensor<AFloat> & );
   static void Softmax(TCudnnTensor<AFloat> &YHat,
                       const TCudnnTensor<AFloat> & );
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

   static AFloat L1Regularization(const TCudnnTensor<AFloat> & W);
   static void AddL1RegularizationGradients(TCudnnTensor<AFloat> & A,
                                            const TCudnnTensor<AFloat> & W,
                                            AFloat weightDecay);

   static AFloat L2Regularization(const TCudnnTensor<AFloat> & W);
   static void AddL2RegularizationGradients(TCudnnTensor<AFloat> & A,
                                            const TCudnnTensor<AFloat> & W,
                                            AFloat weightDecay);
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

   static void InitializeGauss(TCudnnTensor<AFloat> & A);
   static void InitializeUniform(TCudnnTensor<AFloat> & A);
   static void InitializeIdentity(TCudnnTensor<AFloat> & A);
   static void InitializeZero(TCudnnTensor<AFloat> & A);
   static void InitializeGlorotUniform(TCudnnTensor<AFloat> & A);
   static void InitializeGlorotNormal(TCudnnTensor<AFloat> & A);
   // return static instance of random generator used for initialization
   // if generator does not exist it is created the first time with a random seed (e.g. seed = 0)
   static TRandom & GetRandomGenerator(); 
   // set random seed for the static geenrator
   // if the static geneerator does not exists it is created
   static void SetRandomSeed(size_t seed); 


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
   static void Dropout(TCudnnTensor<AFloat> & A, AFloat p);

   ///@}

   //____________________________________________________________________________
   //
   //  Convolutional Layer Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
   ///@{

   /** Attaches a cuda stream to each matrix in order to accomodate parallel kernel launches. */
   static void PrepareInternals(std::vector<TCudnnTensor<AFloat>> & inputPrime);

   /** Calculate how many neurons "fit" in the output layer, given the input as well as the layer's hyperparameters. */
   static size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride);

   /** Add the biases in the Convolutional Layer.  */
   static void AddConvBiases(TCudnnTensor<AFloat> &output, const TCudnnTensor<AFloat> &biases);

   ///@}
   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward();

   /** @name Backward Propagation in Convolutional Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Convolutional Layer. */
   static void ConvLayerBackward();

   /** Utility function for calculating the activation gradients of the layer
    *  before the convolutional layer. */
   static void CalculateConvActivationGradients();

   /** Utility function for calculating the weight gradients of the convolutional
    * layer. */
   static void CalculateConvWeightGradients();

   /** Utility function for calculating the bias gradients of the convolutional
    *  layer */
   static void CalculateConvBiasGradients();

   ///@}

   //____________________________________________________________________________
   //
   //  Max Pooling Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Max Pooling Layer
    */
   ///@{

   /** Downsample the matrix \p C to the matrix \p A, using max
    *  operation, such that the winning indices are stored in matrix
    *  \p B. */
   static void Downsample(TCudnnTensor<AFloat> &A, TCudnnTensor<AFloat> &B, const TCudnnTensor<AFloat> &C,
                          size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth,
                          size_t strideRows, size_t strideCols);
   ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
   ///@{
       
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  winning idices stored in the index matrix, it just forwards the actiovation
    *  gradients to the previous layer. */
   static void MaxPoolLayerBackward();

   ///@}

   //____________________________________________________________________________
   //
   //  Reshape Layer Propagation
   //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
   ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   static void Reshape(TCudnnTensor<AFloat> &A, const TCudnnTensor<AFloat> &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   static void Flatten(TCudnnTensor<AFloat> &A, const std::vector<TCudnnTensor<AFloat>> &B, size_t size, size_t nRows,
                       size_t nCols);

   /** Transforms each row of \p B to a matrix and stores it in the tensor \p B. */
   static void Deflatten(std::vector<TCudnnTensor<AFloat>> &A, const TCudnnTensor<AFloat> &B, size_t index, size_t nRows,
                         size_t nCols);
   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   static void Rearrange(std::vector<TCudnnTensor<AFloat>> &out, const std::vector<TCudnnTensor<AFloat>> &in); 

   ///@}

   //____________________________________________________________________________
   //
   // Optimizers (not in cudnn)
   //____________________________________________________________________________

};

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(TCudaMatrix<AFloat> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<AFloat> tmp = A;
   Copy(B, TCudaMatrix<AFloat>(tmp) ); 
}

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(std::vector<TCudaMatrix<AFloat>> &B,
                            const std::vector<AMatrix_t> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      CopyDiffArch(B[i], A[i]);
   }
}

} // namespace DNN
} // namespace TMVA

#endif
