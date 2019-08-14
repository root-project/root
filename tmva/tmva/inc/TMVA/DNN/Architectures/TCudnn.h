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
#include "Cuda/CudaTensor.h"
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
template<typename AFloat>
class TCudnn
{
private:
   static TRandom * fgRandomGen;
public:

    using Scalar_t       = AFloat;
    using Matrix_t       = TCudaTensor<AFloat>;
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
   //static void Backward();
   
   /** Adds the elements in matrix B scaled by beta to the elements in
    *  the matrix A (scaled by alpha). This is required for the weight update
    *  in the gradient descent step.*/
   static void ScaleAdd(TCudaTensor<Scalar_t> & A,
                        const TCudaTensor<Scalar_t> & B,
                        const Scalar_t alpha = 1.0,
                        const Scalar_t beta  = 1.0);
   /** Copy the elements of matrix A into matrix B. */
   static void Copy(TCudaTensor<AFloat> & B,
                    const TCudaTensor<AFloat> & A);

   // copy from another type of matrix
   //template<typename AMatrix_t>
   //static void CopyDiffArch(TCudaTensor<Scalar_t> & B, const AMatrix_t & A); 


   /** Above functions extended to vectors */
   static void ScaleAdd(std::vector<TCudaTensor<Scalar_t>> & A,
                        const std::vector<TCudaTensor<Scalar_t>> & B,
                        const Scalar_t alpha = 1.0,
                        const Scalar_t beta  = 1.0);

   static void Copy(std::vector<TCudaTensor<Scalar_t>> & A,
                    const std::vector<TCudaTensor<Scalar_t>> & B);

   // copy from another architecture
   /*template<typename AMatrix_t>
   static void CopyDiffArch(std::vector<TCudaTensor<Scalar_t>> & A,
const std::vector<AMatrix_t> & B);*/

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
   static void Identity(TCudaTensor<AFloat> & B);
   /*static void IdentityDerivative(TCudaTensor<AFloat> & B,
                                  const TCudaTensor<AFloat> & A);*/

   static void Activation(TCudaTensor<AFloat> & B, EActivationFunction activFunct,
                          const double coefRelu = 0.0, const AFloat alpha = 1, const AFloat beta = 1);
                    
   static void Relu(TCudaTensor<AFloat> & B, const double coefRelu = 0.0, 
                    const AFloat alpha = 1, const AFloat beta = 1);
                    
   /*static void ReluDerivative(TCudaTensor<AFloat> & B,
                              const TCudaTensor<AFloat> & A);*/

   static void Sigmoid(TCudaTensor<AFloat> & B, const double coefRelu = 0.0, 
                    const AFloat alpha = 1, const AFloat beta = 1);
   /*static void SigmoidDerivative(TCudaTensor<AFloat> & B,
                                 const TCudaTensor<AFloat> & A);*/

   static void Tanh(TCudaTensor<AFloat> & B, const double coefRelu = 0.0, 
                    const AFloat alpha = 1, const AFloat beta = 1);
   /*static void TanhDerivative(TCudaTensor<AFloat> & B,
                              const TCudaTensor<AFloat> & A);*/

   //static void SymmetricRelu(TCudaTensor<AFloat> & B);
   /*static void SymmetricReluDerivative(TCudaTensor<AFloat> & B,
                                       const TCudaTensor<AFloat> & A);*/

   //static void SoftSign(TCudaTensor<AFloat> & B);
   /*static void SoftSignDerivative(TCudaTensor<AFloat> & B,
                                  const TCudaTensor<AFloat> & A);*/

   //static void Gauss(TCudaTensor<AFloat> & B);
   /*static void GaussDerivative(TCudaTensor<AFloat> & B,
                               const TCudaTensor<AFloat> & A);*/
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

   /*static AFloat MeanSquaredError(const TCudaTensor<AFloat> &Y, const TCudaTensor<AFloat> &output,
                                  const TCudaTensor<AFloat> &weights);
   static void MeanSquaredErrorGradients(TCudaTensor<AFloat> &dY, const TCudaTensor<AFloat> &Y,
                                         const TCudaTensor<AFloat> &output, const TCudaTensor<AFloat> &weights);*/

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   /*static AFloat CrossEntropy(const TCudaTensor<AFloat> &Y, const TCudaTensor<AFloat> &output,
                              const TCudaTensor<AFloat> &weights);

   static void CrossEntropyGradients(TCudaTensor<AFloat> &dY, const TCudaTensor<AFloat> &Y,
                                     const TCudaTensor<AFloat> &output, const TCudaTensor<AFloat> &weights);*/

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   /*static AFloat SoftmaxCrossEntropy(const TCudaTensor<AFloat> &Y, const TCudaTensor<AFloat> &output,
                                     const TCudaTensor<AFloat> &weights);
   static void SoftmaxCrossEntropyGradients(TCudaTensor<AFloat> &dY, const TCudaTensor<AFloat> &Y,
                                            const TCudaTensor<AFloat> &output, const TCudaTensor<AFloat> &weights);*/
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
   /*static void Sigmoid(TCudaTensor<AFloat> &YHat,
                       const TCudaTensor<AFloat> & );
   static void Softmax(TCudaTensor<AFloat> &YHat,
                       const TCudaTensor<AFloat> & );*/
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

   /*static AFloat L1Regularization(const TCudaTensor<AFloat> & W);
   static void AddL1RegularizationGradients(TCudaTensor<AFloat> & A,
                                            const TCudaTensor<AFloat> & W,
                                            AFloat weightDecay);

   static AFloat L2Regularization(const TCudaTensor<AFloat> & W);
   static void AddL2RegularizationGradients(TCudaTensor<AFloat> & A,
                                            const TCudaTensor<AFloat> & W,
                                            AFloat weightDecay);*/
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

   /*static void InitializeGauss(TCudaTensor<AFloat> & A);
   static void InitializeUniform(TCudaTensor<AFloat> & A);
   static void InitializeIdentity(TCudaTensor<AFloat> & A);
   static void InitializeZero(TCudaTensor<AFloat> & A);
   static void InitializeGlorotUniform(TCudaTensor<AFloat> & A);
   static void InitializeGlorotNormal(TCudaTensor<AFloat> & A);
   // return static instance of random generator used for initialization
   // if generator does not exist it is created the first time with a random seed (e.g. seed = 0)
   static TRandom & GetRandomGenerator(); 
   // set random seed for the static geenrator
   // if the static geneerator does not exists it is created
   static void SetRandomSeed(size_t seed); */


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
   //static void Dropout(TCudaTensor<AFloat> & A, AFloat p);

   ///@}


   //____________________________________________________________________________
   //
   //  Convolutional Layer Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
   ///@{

   /** Attaches a cuda stream to each matrix in order to accomodate parallel kernel launches. */
   static void PrepareInternals(std::vector<TCudaTensor<AFloat>> & inputPrime, 
                                cudnnFilterDescriptor_t filterDescr = nullptr,
                                cudnnConvolutionDescriptor_t fConvolutionDescriptor = nullptr);

   /** Calculate how many neurons "fit" in the output layer, given the input as well as the layer's hyperparameters. */
   //static size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride);

   /** Transform the matrix \p B in local view format, suitable for
    *  convolution, and store it in matrix \p A. */
   /*static void Im2col(TCudaTensor<AFloat> &A,
                      const TCudaTensor<AFloat> &B,
                      size_t imgHeight,
                      size_t imgWidth,
                      size_t fltHeight,
                      size_t fltWidth,
                      size_t strideRows,
                      size_t strideCols,
                      size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);*/

   //static void Im2colIndices(std::vector<int> & /* V */, const TCudaTensor<AFloat> & /* B */, size_t /* nLocalViews */,
   //                          size_t /* imgHeight */, size_t /* imgWidth */, size_t /* fltHeight */,
   //                          size_t /* fltWidth */, size_t /* strideRows */, size_t /* strideCols */,
   //                          size_t /* zeroPaddingHeight */, size_t /* zeroPaddingWidth */) {}
   //static void Im2colFast(TCudaTensor<AFloat> & /* A */, const TCudaTensor<AFloat> & /* B */,
   //                       const std::vector<int> & /* V */) {}


   /** Rotates the matrix \p B, which is representing a weights,
    *  and stores them in the matrix \p A. */
   /*static void RotateWeights(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> &B, size_t filterDepth,
                             size_t filterHeight, size_t filterWidth, size_t numFilters);*/

   /** Add the biases in the Convolutional Layer.  */
   //static void AddConvBiases(TCudaTensor<AFloat> &output, const TCudaTensor<AFloat> &biases);

   /** Set TCudaTensor as cuDNN Filter */
   static void ConvertToFilter(TCudaTensor<AFloat> &weightsTensor, cudnnFilterDescriptor_t &filter);
   
   ///@}
   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward(std::vector<TCudaTensor<AFloat>> & output,
                                std::vector<TCudaTensor<AFloat>> & derivatives,
                                const std::vector<TCudaTensor<AFloat>> &input,
                                const TCudaTensor<AFloat> &weights, const TCudaTensor<AFloat> & biases,
                                const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                std::vector<TCudaTensor<AFloat>> & inputPrime,
                                const AFloat alpha = 1,
                                const AFloat beta  = 1);

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
   /*static void Downsample(TCudaTensor<AFloat> &A, TCudaTensor<AFloat> &B, const TCudaTensor<AFloat> &C,
                          size_t imgHeight, size_t imgWidth, size_t fltHeight, size_t fltWidth,
                          size_t strideRows, size_t strideCols);*/
   ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
   ///@{
       
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  winning idices stored in the index matrix, it just forwards the actiovation
    *  gradients to the previous layer. */
   //static void MaxPoolLayerBackward();

   ///@}

   //____________________________________________________________________________
   //
   //  Reshape Layer Propagation
   //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
   ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   //static void Reshape(TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   /*static void Flatten(TCudaTensor<AFloat> &A, const std::vector<TCudaTensor<AFloat>> &B, size_t size, size_t nRows,
                       size_t nCols);*/

   /** Transforms each row of \p B to a matrix and stores it in the tensor \p B. */
   /*static void Deflatten(std::vector<TCudaTensor<AFloat>> &A, const TCudaTensor<AFloat> &B, size_t index, size_t nRows,
                         size_t nCols);*/
   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   //static void Rearrange(std::vector<TCudaTensor<AFloat>> &out, const std::vector<TCudaTensor<AFloat>> &in); 

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
   static void Multiply(TCudaTensor<Scalar_t> & C,
                        const TCudaTensor<Scalar_t> & A,
                        const TCudaTensor<Scalar_t> & B,
                        const Scalar_t alpha = 1.0,
                        const Scalar_t beta  = 1.0,
                        const Scalar_t gamma = 0.0);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   /*static void TransposeMultiply(TCudaTensor<AFloat> & output,
                                 const TCudaTensor<AFloat> & input,
                                 const TCudaTensor<AFloat> & Weights);*/
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   /*static void Hadamard(TCudaTensor<AFloat> & A, const TCudaTensor<AFloat> & B);*/

   /** Sum columns of (m x n) matrix \p A and write the results into the first
    * m elements in \p B.
    */
   /*static void SumColumns(TCudaTensor<AFloat> & B, const TCudaTensor<AFloat> & A);*/

   /** Sum rows of (m x n) matrix \p A and write the results into the first
   * m elements in \p B.
   */
   /*static void SumRows(TCudaTensor<AFloat> & B, const TCudaTensor<AFloat> & A);*/

   /** Compute the sum of all elements in \p A */
   static AFloat Sum(const TCudaTensor<Scalar_t> &A, const Scalar_t alpha = 1.0, const Scalar_t beta = 0.0);
   
   /** Extend the sum of a CudaTensor to a vector  */
   static AFloat Sum(const std::vector<TCudaTensor<Scalar_t> > &A, 
                     const Scalar_t alpha = 1.0, const Scalar_t beta = 0.0);

   /** Check two matrices for equality, taking floating point arithmetic errors into account. */
   static bool AlmostEquals(const TCudaTensor<AFloat> &A, const TCudaTensor<AFloat> &B, double epsilon = 0.1);

   /** Add the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstAdd(TCudaTensor<Scalar_t> &A, const Scalar_t beta);

   /** Multiply the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstMult(TCudaTensor<Scalar_t> &A, const Scalar_t beta);

   /** Reciprocal each element of the matrix \p A and write the result into
    * \p A
    */
   /*static void ReciprocalElementWise(TCudaTensor<AFloat> &A);*/

   /** Square each element of the matrix \p A and write the result into
    * \p A
    */
   /*static void SquareElementWise(TCudaTensor<AFloat> &A);*/

   /** Square root each element of the matrix \p A and write the result into
    * \p A
    */
   static void SqrtElementWise(TCudaTensor<Scalar_t> &A, const Scalar_t alpha = 1, const Scalar_t beta = 0, const Scalar_t gamma = 0);

   //____________________________________________________________________________
   //
   // Optimizers (not in cudnn)
   //____________________________________________________________________________
   
   // optimizer functions
   /*static void AdamUpdate(TCudaTensor<AFloat> & A, const TCudaTensor<AFloat> & M, const TCudaTensor<AFloat> & V, AFloat alpha, AFloat eps);
   static void AdamUpdateFirstMom(TCudaTensor<AFloat> & A, const TCudaTensor<AFloat> & B, AFloat beta);
   static void AdamUpdateSecondMom(TCudaTensor<AFloat> & A, const TCudaTensor<AFloat> & B, AFloat beta);*/

};

//____________________________________________________________________________
/*template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(TCudaTensor<AFloat> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<AFloat> tmp = A;
   Copy(B, TCudaTensor<AFloat>(tmp) ); 
}

//____________________________________________________________________________
template <typename AFloat>
template <typename AMatrix_t>
void TCuda<AFloat>::CopyDiffArch(std::vector<TCudaTensor<AFloat>> &B,
                            const std::vector<AMatrix_t> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      CopyDiffArch(B[i], A[i]);
   }
}*/

} // namespace DNN
} // namespace TMVA

#endif
