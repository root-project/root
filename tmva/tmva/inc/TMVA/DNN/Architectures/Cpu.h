// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 05/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
// Definition of the TCpu architecture, which provides a         //
 // multi-threaded CPU implementation of the low-level interface //
 // networks for Cpus using BLAS and Roots TThreadExecutor            //
 //////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_CPU
#define TMVA_DNN_ARCHITECTURES_CPU

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/CNN/ConvLayer.h"

#include "Cpu/CpuBuffer.h"
#include "Cpu/CpuMatrix.h"
#include <vector>

class TRandom;

namespace TMVA
{
namespace DNN
{
   //class EActivationFunction;

 struct DummyFilterDescriptor {};
 struct DummyConvolutionDescriptor {}; 

/** The TCpu architecture class.
 *
 * Low-level interface class for multi-threaded CPU architectures. Contains as
 * public types the declaration of the scalar, matrix and data loader types
 * for this architecture as well as the remaining functions in the low-level
 * interface in the form of static members.
 */
template<typename AReal = Real_t>
class TCpu
{
private:
   static TRandom * fgRandomGen;
public:

   using Scalar_t       = AReal;
   using Tensor_t       = TCpuTensor<AReal>;
   using Matrix_t       = TCpuMatrix<AReal>;
   using HostBuffer_t   = TCpuBuffer<AReal>;
   using DeviceBuffer_t = TCpuBuffer<AReal>;

   using ConvolutionDescriptor_t = DummyConvolutionDescriptor;
   using FilterDescriptor_t = DummyFilterDescriptor; 


   // // Utility function to convert from a Matrix to a Tensor
   // static Tensor_t  MatrixToTensor(Matrix_t & A) { 
   //    return Tensor_t(A.GetRawDataPointer(), Shape_t({A.GetNrows(), A.GetNcols() }), RTensor::MemoryLayout::ColumnMajor);
   // }

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
   static void MultiplyTranspose(Matrix_t &output, const Matrix_t &input, const Matrix_t &weights);

   static void MultiplyTranspose(Tensor_t &output, const Tensor_t &input, const Matrix_t &weights) {
      Matrix_t output_matrix = output.GetMatrix(); 
      MultiplyTranspose( output_matrix, input.GetMatrix(), weights);
      //ensor_t::MatrixToTensor(output_matrix, output); // this maybe is not needed
   }

   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(Matrix_t &output,const Matrix_t &biases);

   static void AddRowWise(Tensor_t &output, const Matrix_t &biases) { 
      Matrix_t output_matrix = output.GetMatrix(); 
      AddRowWise(output_matrix, biases); 
      //Tensor_t::MatrixToTensor(output_matrix, output); // this maybe is not needed
   }

   /** @name Backward Propagation (Dense Layers)
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
   static void Backward(Tensor_t & activationGradientsBackward,
                        Matrix_t & weightGradients,
                        Matrix_t & biasGradients,
                        Tensor_t & df,
                        const Tensor_t & activationGradients,
                        const Matrix_t & weights,
                        const Tensor_t & activationBackward);

  



   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(Matrix_t & A,
                        const Matrix_t & B,
                        Scalar_t beta = 1.0);

   static void Copy(Matrix_t & B,
                    const Matrix_t & A);

   // copy from another type of matrix
   template<typename AMatrix_t>
   static void CopyDiffArch(Matrix_t & B, const AMatrix_t & A); 


   /** Above functions extended to vectors */
   static void ScaleAdd(Tensor_t & A,
                        const Tensor_t & B,
                        Scalar_t beta = 1.0);

   static void Copy(Tensor_t & A,
                    const Tensor_t & B);

   // copy from another tensor
   template<typename ATensor_t>
   static void CopyDiffArch(Tensor_t & A,
                     const ATensor_t & B);

   // copy from vector of matrices of different types
   template<typename AMatrix_t>
   static void CopyDiffArch(std::vector<Matrix_t>  & A,
                      const std::vector<AMatrix_t> & B);

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
   static void IdentityDerivative(Tensor_t & B,
                                  const Tensor_t &A);

   static void Relu(Tensor_t & B);
   static void ReluDerivative(Tensor_t & B,
                              const Tensor_t & A);

   static void Sigmoid(Tensor_t & B);
   static void SigmoidDerivative(Tensor_t & B,
                                 const Tensor_t & A);

   static void Tanh(Tensor_t & B);
   static void TanhDerivative(Tensor_t & B,
                              const Tensor_t & A);

   static void SymmetricRelu(Tensor_t & B);
   static void SymmetricReluDerivative(Tensor_t & B,
                                       const Tensor_t & A);

   static void SoftSign(Tensor_t & B);
   static void SoftSignDerivative(Tensor_t & B,
                                  const Tensor_t & A);

   static void Gauss(Tensor_t & B);
   static void GaussDerivative(Tensor_t & B,
                               const Tensor_t & A);
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

   static Scalar_t MeanSquaredError(const Matrix_t &Y, const Matrix_t &output,
                                    const Matrix_t &weights);
   static void MeanSquaredErrorGradients(Matrix_t &dY, const Matrix_t &Y,
                                         const Matrix_t &output, const Matrix_t &weights);

   /** Sigmoid transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t CrossEntropy(const Matrix_t &Y, const Matrix_t &output,
                                const Matrix_t &weights);

   static void CrossEntropyGradients(Matrix_t &dY, const Matrix_t &Y,
                                     const Matrix_t &output, const Matrix_t &weights);

   /** Softmax transformation is implicitly applied, thus \p output should
    *  hold the linear activations of the last layer in the net. */
   static Scalar_t SoftmaxCrossEntropy(const Matrix_t &Y, const Matrix_t &output,
                                       const Matrix_t &weights);
   static void SoftmaxCrossEntropyGradients(Matrix_t &dY, const Matrix_t &Y,
                                            const Matrix_t &output, const Matrix_t &weights);
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
   static void Sigmoid(Matrix_t &YHat,
                        const Matrix_t & );
   static void Softmax(Matrix_t &YHat,
                       const Matrix_t & );
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

   static Scalar_t L1Regularization(const Matrix_t & W);
   static void AddL1RegularizationGradients(Matrix_t & A,
                                            const Matrix_t & W,
                                            Scalar_t weightDecay);

   static Scalar_t L2Regularization(const Matrix_t & W);
   static void AddL2RegularizationGradients(Matrix_t & A,
                                            const Matrix_t & W,
                                            Scalar_t weightDecay);
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

   static void InitializeGauss(Matrix_t & A);
   static void InitializeUniform(Matrix_t & A);
   static void InitializeIdentity(Matrix_t & A);
   static void InitializeZero(Matrix_t & A);
   static void InitializeGlorotNormal(Matrix_t & A);
   static void InitializeGlorotUniform(Matrix_t & A);

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
    *  tensor \p A and scale the result by reciprocal of \p p. */
   static void Dropout(Tensor_t & A, Scalar_t p);

   static void Dropout(Matrix_t & A, Scalar_t p) { 
      Tensor_t tA(A);
      Dropout( tA, p );
   }

   ///@}

   //____________________________________________________________________________
   //
   //  Convolutional Layer Propagation
   //____________________________________________________________________________

   /** @name Forward Propagation in Convolutional Layer
    */
   ///@{

   /** Calculate how many neurons "fit" in the output layer, given the input as well as the layer's hyperparameters. */
   static size_t calculateDimension(size_t imgDim, size_t fltDim, size_t padding, size_t stride);

   /** Transform the matrix B in local view format, suitable for
    *  convolution, and store it in matrix A */
   static void Im2col(Matrix_t &A,
                      const Matrix_t &B,
                      size_t imgHeight,
                      size_t imgWidth,
                      size_t fltHeight,
                      size_t fltWidth,
                      size_t strideRows,
                      size_t strideCols,
                      size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);

   static void Im2colIndices(std::vector<int> &V, const Matrix_t &B, size_t nLocalViews, size_t imgHeight, size_t imgWidth, size_t fltHeight,
                      size_t fltWidth, size_t strideRows, size_t strideCols, size_t zeroPaddingHeight,
                      size_t zeroPaddingWidth);
   static void Im2colFast(Matrix_t &A, const Matrix_t &B, const std::vector<int> & V); 

   /** Rotates the matrix \p B, which is representing a weights,
    *  and stores them in the matrix \p A. */
   static void RotateWeights(Matrix_t &A, const Matrix_t &B, size_t filterDepth, size_t filterHeight,
                             size_t filterWidth, size_t numFilters);

   /** Add the biases in the Convolutional Layer.  */
   static void AddConvBiases(Matrix_t &output, const Matrix_t &biases);
   ///@}

   /** Dummy placeholder - preparation is currently only required for the CUDA architecture. */
   static void PrepareInternals(Tensor_t &) {}

   /** Forward propagation in the Convolutional layer */
   static void ConvLayerForward(Tensor_t & output,
                                Tensor_t & derivatives,
                                const Tensor_t &input,
                                const Matrix_t &weights, const Matrix_t & biases,
                                const DNN::CNN::TConvParams & params, EActivationFunction activFunc,
                                Tensor_t & /* inputPrime */);

   /** @name Backward Propagation in Convolutional Layer
    */
   ///@{

   /** Perform the complete backward propagation step in a Convolutional Layer.
    *  If the provided \p activationGradientsBackward matrix is not empty, compute the
    *  gradients of the objective function with respect to the activations
    *  of the previous layer (backward direction).
    *  Also compute the weight and the bias gradients. Modifies the values
    *  in \p df and thus produces only a valid result, if it is applied the
    *  first time after the corresponding forward propagation has been per-
    *  formed. */
   static void ConvLayerBackward(Tensor_t &activationGradientsBackward,
                                 Matrix_t &weightGradients, Matrix_t &biasGradients,
                                 Tensor_t &df,
                                 const Tensor_t &activationGradients,
                                 const Matrix_t &weights,
                                 const Tensor_t &activationBackward, size_t batchSize,
                                 size_t inputHeight, size_t inputWidth, size_t depth, size_t height, size_t width,
                                 size_t filterDepth, size_t filterHeight, size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the activation gradients of the layer
    *  before the convolutional layer. */
   static void CalculateConvActivationGradients(Tensor_t &activationGradientsBackward,
                                                const Tensor_t &df,
                                                const Matrix_t &weights, size_t batchSize,
                                                size_t inputHeight, size_t inputWidth, size_t depth, size_t height,
                                                size_t width, size_t filterDepth, size_t filterHeight,
                                                size_t filterWidth);

   /** Utility function for calculating the weight gradients of the convolutional
    * layer. */
   static void CalculateConvWeightGradients(Matrix_t &weightGradients,
                                            const Tensor_t &df,
                                            const Tensor_t &activations_backward,
                                            size_t batchSize, size_t inputHeight, size_t inputWidth, size_t depth,
                                            size_t height, size_t width, size_t filterDepth, size_t filterHeight,
                                            size_t filterWidth, size_t nLocalViews);

   /** Utility function for calculating the bias gradients of the convolutional
    *  layer */
   static void CalculateConvBiasGradients(Matrix_t &biasGradients, const Tensor_t &df,
                                          size_t batchSize, size_t depth, size_t nLocalViews);
   ///@}

   //____________________________________________________________________________
   //
   //  Max Pooling Layer Propagation
   //____________________________________________________________________________
   /** @name Forward Propagation in Max Pooling Layer
    */
   ///@{

   /** Downsample the matrix \p C to the matrix \p A, using max
    * operation, such that the winning indices are stored in matrix
    * \p B. */
   static void Downsample(Tensor_t &A, Tensor_t &B, const Tensor_t &C, size_t imgHeight,
                          size_t imgWidth, size_t fltHeight, size_t fltWidth, size_t strideRows, size_t strideCols);

   ///@}

   /** @name Backward Propagation in Max Pooling Layer
    */
   ///@{
   /** Perform the complete backward propagation step in a Pooling Layer. Based on the
    *  winning idices stored in the index matrix, it just forwards the actiovation
    *  gradients to the previous layer. */
   static void MaxPoolLayerBackward(Tensor_t &activationGradientsBackward,
                                    const Tensor_t &activationGradients,
                                    const Tensor_t &indexMatrix,
                                    size_t imgHeight,
                                    size_t imgWidth,
                                    size_t fltHeight,
                                    size_t fltWidth,
                                    size_t strideRows,
                                    size_t strideCols,
                                    size_t nLocalViews);

   ///@}

   //____________________________________________________________________________
   //
   //  Reshape Layer Propagation
   //____________________________________________________________________________
   /** @name Forward and Backward Propagation in Reshape Layer
    */
   ///@{

   /** Transform the matrix \p B to a matrix with different dimensions \p A */
   static void Reshape(Matrix_t &A, const Matrix_t &B);

   /** Flattens the tensor \p B, such that each matrix, is stretched in
    *  one row, resulting with a matrix \p A. */
   static void Flatten(Tensor_t &A, const Tensor_t &B); // size_t size, size_t nRows, size_t nCols);

   /** Transforms each row of \p B to a matrix and stores it in the
    *  tensor \p B. */
   static void Deflatten(Tensor_t &A, const Tensor_t &B); // size_t index, size_t nRows,size_t nCols);

   /** Rearrage data accoring to time fill B x T x D out with T x B x D matrix in*/
   static void Rearrange(Tensor_t &out, const Tensor_t &in); 


 /** Backward pass for Recurrent Networks */
   static Matrix_t & RecurrentLayerBackward(Matrix_t & state_gradients_backward, // BxH
                                            Matrix_t & input_weight_gradients,
                                            Matrix_t & state_weight_gradients,
                                            Matrix_t & bias_gradients,
                                            Matrix_t & df, //DxH
                                            const Matrix_t & state, // BxH
                                            const Matrix_t & weights_input, // HxD 
                                            const Matrix_t & weights_state, // HxH
                                            const Matrix_t & input,  // BxD
                                            Matrix_t & input_gradient);


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
   static void Multiply(Matrix_t &C,
                        const Matrix_t &A,
                        const Matrix_t &B);
   /** Matrix multiplication of two matrices \p A and \p B^T (transposed) with the
    *  result being written into C.
    */
   static void TransposeMultiply(Matrix_t &output,
                                 const Matrix_t &input,
                                 const Matrix_t &Weights,
                                 Scalar_t alpha = 1.0, Scalar_t beta = 0.);
   /** In-place Hadamard (element-wise) product of matrices \p A and \p B
    *  with the result being written into \p A.
    */
   static void Hadamard(Tensor_t &A,
                        const Tensor_t &B);
   static void Hadamard(Matrix_t &A,
                        const Matrix_t &B);
   // {
   //    Tensor_t tA(A);
   //    Hadamard( tA, Tensor_t(B));
   // }

   /** Sum columns of (m x n) matrixx \p A and write the results into the first
    * m elements in \p A.
    */
   static void SumColumns(Matrix_t &B,
                          const Matrix_t &A,
                          Scalar_t alpha = 1.0, Scalar_t beta = 0.);

   /** Compute the sum of all elements in \p A */
   static Scalar_t Sum(const Matrix_t &A);

   /** Check two matrices for equality, taking floating point arithmetic errors into account. */
   static bool AlmostEquals(const Matrix_t &A, const Matrix_t &B, double epsilon = 0.1);

   /** Add the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstAdd(Matrix_t &A, Scalar_t beta);

   /** Multiply the constant \p beta to all the elements of matrix \p A and write the
    * result into \p A.
    */
   static void ConstMult(Matrix_t &A, Scalar_t beta);

   /** Reciprocal each element of the matrix \p A and write the result into
    * \p A
    */
   static void ReciprocalElementWise(Matrix_t &A);

   /** Square each element of the matrix \p A and write the result into
    * \p A
    */
   static void SquareElementWise(Matrix_t &A);

   /** Square root each element of the matrix \p A and write the result into
    * \p A
    */
   static void SqrtElementWise(Matrix_t &A);

     // optimizer functions
   static void AdamUpdate(Matrix_t & A, const Matrix_t & M, const Matrix_t & V, Scalar_t alpha, Scalar_t eps);
   static void AdamUpdateFirstMom(Matrix_t & A, const Matrix_t & B, Scalar_t beta);
   static void AdamUpdateSecondMom(Matrix_t & A, const Matrix_t & B, Scalar_t beta);

   // printing of tensor
   static void PrintTensor( const Tensor_t & A, const std::string name = "tensor");

};

//____________________________________________________________________________
template <typename Real_t>
template <typename AMatrix_t>
void TCpu<Real_t>::CopyDiffArch(TCpuMatrix<Real_t> &B,
                        const AMatrix_t &A)
{
   // copy from another architecture using the reference one
   // this is not very efficient since creates temporary objects
   TMatrixT<Real_t> tmp = A;
   Copy(B, TCpuMatrix<Real_t>(tmp) ); 
}

//____________________________________________________________________________
template <typename Real_t>
template <typename ATensor_t>
void TCpu<Real_t>::CopyDiffArch(TCpuTensor<Real_t> &B,
                            const ATensor_t &A)
{
   // assume tensor are dim 3 and can be converted in a matrix
   for (size_t i = 0; i < B.GetFirstSize(); ++i) {
      CopyDiffArch(B.At(i).GetMatrix(), A.At(i).GetMatrix());
   }
}

// implementation using vector of matrices for the weights
template <typename Real_t>
template <typename AMatrix_t>
void TCpu<Real_t>::CopyDiffArch(std::vector<TCpuMatrix<Real_t>> &A, const std::vector<AMatrix_t> &B)
{
   for (size_t i = 0; i < A.size(); ++i) {
      CopyDiffArch(A[i], B[i]);
   }
}

template <typename Real_t>
void TCpu<Real_t>::PrintTensor(const typename TCpu<Real_t>::Tensor_t & A, const std::string name ) 
{
   std::cout << name << " tensor size = " << A.GetSize() << " shape = { "; 
   auto shape = A.GetShape(); 
   for (size_t k = 0; k < shape.size()-1; ++k)
      std::cout << shape[k] << " , ";
   std::cout << shape.back() << " } ";
   //std::cout << " is view/owner " << A.IsView() << " / " << A.IsOwner() << " layout " << A.GetLayout() << std::endl;
   //std::cout << " layout " << A.GetLayout() << std::endl;
   // print elements 
   // need to find way to nice printing all elements
   std::cout << " tensor count " << A.GetBufferUseCount() << std::endl;
   if (A.GetShape().size() == 2 ) { 
      for (size_t i = 0; i < A.GetShape()[0]; ++i) {
         std::cout << "{ ";
         for (size_t j = 0; j < A.GetShape()[1]; ++j) {
            std::cout << A(i,j) << " ";
         }
         std::cout << " } " << std::endl;      
      }
   } else if  (A.GetShape().size() == 3 ) {
      for (size_t i = 0; i < A.GetFirstSize(); ++i) {
         std::cout << "{ ";
         for (size_t j = 0; j < A.GetHSize(); ++j) {
            std::cout << "{ ";
            for (size_t k = 0; k < A.GetWSize(); ++k) {
               std::cout << A(i,j,k) << " ";
            }
            std::cout << " } " << std::endl;
         }
         std::cout << " } " << std::endl;
      }    
   }
   else {  
      for (size_t l = 0; l < A.GetSize(); ++l) {
         std::cout << A.GetData()[l] << " ";
      }
   }  
   std::cout << "\n";
   std::cout << "********\n";
}




} // namespace DNN
} // namespace TMVA

#endif
