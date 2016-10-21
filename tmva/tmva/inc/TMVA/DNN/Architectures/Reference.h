// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////////
// Declaration of the TReference architecture, which provides a      //
// reference implementation of the low-level interface for the DNN   //
// implementation based on ROOT's TMatrixT matrix type.              //
///////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_ARCHITECTURES_REFERENCE
#define TMVA_DNN_ARCHITECTURES_REFERENCE

#include "TMatrix.h"

namespace TMVA
{
namespace DNN
{

/*! The reference architecture class.
*
* Class template that contains the reference implementation of the low-level
* interface for the DNN implementation. The reference implementation uses the
* TMatrixT class template to represent matrices.
*
* \tparam AReal The floating point type used to represent scalars.
*/
template<typename AReal>
class TReference
{
public:

   using Scalar_t     = AReal;
   using Matrix_t     = TMatrixT<AReal>;

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
   static void MultiplyTranspose(TMatrixT<Scalar_t> &output,
                                 const TMatrixT<Scalar_t> &input,
                                 const TMatrixT<Scalar_t> &weights);
   /** Add the vectors biases row-wise to the matrix output */
   static void AddRowWise(TMatrixT<Scalar_t> &output,
                          const TMatrixT<Scalar_t> &biases);
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
   static void Backward(TMatrixT<Scalar_t> & activationGradientsBackward,
                        TMatrixT<Scalar_t> & weightGradients,
                        TMatrixT<Scalar_t> & biasGradients,
                        TMatrixT<Scalar_t> & df,
                        const TMatrixT<Scalar_t> & activationGradients,
                        const TMatrixT<Scalar_t> & weights,
                        const TMatrixT<Scalar_t> & activationBackward);
   /** Adds a the elements in matrix B scaled by c to the elements in
    *  the matrix A. This is required for the weight update in the gradient
    *  descent step.*/
   static void ScaleAdd(TMatrixT<Scalar_t> & A,
                        const TMatrixT<Scalar_t> & B,
                        Scalar_t beta = 1.0);

   static void Copy(TMatrixT<Scalar_t> & A,
                    const TMatrixT<Scalar_t> & B);
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
   static void Identity(TMatrixT<AReal> & B);
   static void IdentityDerivative(TMatrixT<AReal> & B,
                                  const TMatrixT<AReal> & A);

   static void Relu(TMatrixT<AReal> & B);
   static void ReluDerivative(TMatrixT<AReal> & B,
                              const TMatrixT<AReal> & A);

   static void Sigmoid(TMatrixT<AReal> & B);
   static void SigmoidDerivative(TMatrixT<AReal> & B,
                                 const TMatrixT<AReal> & A);

   static void Tanh(TMatrixT<AReal> & B);
   static void TanhDerivative(TMatrixT<AReal> & B,
                              const TMatrixT<AReal> & A);

   static void SymmetricRelu(TMatrixT<AReal> & B);
   static void SymmetricReluDerivative(TMatrixT<AReal> & B,
                                       const TMatrixT<AReal> & A);

   static void SoftSign(TMatrixT<AReal> & B);
   static void SoftSignDerivative(TMatrixT<AReal> & B,
                                  const TMatrixT<AReal> & A);

   static void Gauss(TMatrixT<AReal> & B);
   static void GaussDerivative(TMatrixT<AReal> & B,
                               const TMatrixT<AReal> & A);

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

   static AReal MeanSquaredError(const TMatrixT<AReal> &Y,
                                  const TMatrixT<AReal> &output);
   static void MeanSquaredErrorGradients(TMatrixT<AReal> & dY,
                                         const TMatrixT<AReal> &Y,
                                         const TMatrixT<AReal> &output);

    /** Sigmoid transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static AReal CrossEntropy(const TMatrixT<AReal> &Y,
                              const TMatrixT<AReal> &output);

   static void CrossEntropyGradients(TMatrixT<AReal> & dY,
                                     const TMatrixT<AReal> & Y,
                                     const TMatrixT<AReal> & output);

    /** Softmax transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static AReal SoftmaxCrossEntropy(const TMatrixT<AReal> &Y,
                                    const TMatrixT<AReal> &output);
   static void SoftmaxCrossEntropyGradients(TMatrixT<AReal> & dY,
                                            const TMatrixT<AReal> & Y,
                                            const TMatrixT<AReal> & output);
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
   static void Sigmoid(TMatrixT<AReal> &YHat,
                       const TMatrixT<AReal> & );
   static void Softmax(TMatrixT<AReal> &YHat,
                       const TMatrixT<AReal> & );
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

   static AReal L1Regularization(const TMatrixT<AReal> & W);
   static void AddL1RegularizationGradients(TMatrixT<AReal> & A,
                                            const TMatrixT<AReal> & W,
                                            AReal weightDecay);

   static AReal L2Regularization(const TMatrixT<AReal> & W);
   static void AddL2RegularizationGradients(TMatrixT<AReal> & A,
                                            const TMatrixT<AReal> & W,
                                            AReal weightDecay);
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

   static void InitializeGauss(TMatrixT<AReal> & A);

   static void InitializeUniform(TMatrixT<AReal> & A);

   static void InitializeIdentity(TMatrixT<AReal> & A);

   static void InitializeZero(TMatrixT<AReal> & A);

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
   static void Dropout(TMatrixT<AReal> & A, AReal dropoutProbability);

   ///@}
};

} // namespace DNN
} // namespace TMVA

#endif
