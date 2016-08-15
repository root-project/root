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
#include "Reference/DataLoader.h"

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
* \tparam Real_t The floating point type used to represent scalars.
*/
template<typename Real_t>
class TReference
{
public:

   using Scalar_t     = Real_t;
   using Matrix_t     = TMatrixT<Real_t>;
   template <typename Data_t>
   using DataLoader_t = TReferenceDataLoader<Data_t, Real_t>;

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
   static void Identity(TMatrixT<Real_t> & B);
   static void IdentityDerivative(TMatrixT<Real_t> & B,
                                  const TMatrixT<Real_t> & A);

   static void Relu(TMatrixT<Real_t> & B);
   static void ReluDerivative(TMatrixT<Real_t> & B,
                              const TMatrixT<Real_t> & A);

   static void Sigmoid(TMatrixT<Real_t> & B);
   static void SigmoidDerivative(TMatrixT<Real_t> & B,
                                 const TMatrixT<Real_t> & A);

   static void Tanh(TMatrixT<Real_t> & B);
   static void TanhDerivative(TMatrixT<Real_t> & B,
                              const TMatrixT<Real_t> & A);

   static void SymmetricRelu(TMatrixT<Real_t> & B);
   static void SymmetricReluDerivative(TMatrixT<Real_t> & B,
                                       const TMatrixT<Real_t> & A);

   static void SoftSign(TMatrixT<Real_t> & B);
   static void SoftSignDerivative(TMatrixT<Real_t> & B,
                                  const TMatrixT<Real_t> & A);

   static void Gauss(TMatrixT<Real_t> & B);
   static void GaussDerivative(TMatrixT<Real_t> & B,
                               const TMatrixT<Real_t> & A);

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

   static Real_t MeanSquaredError(const TMatrixT<Real_t> &Y,
                                  const TMatrixT<Real_t> &output);
   static void MeanSquaredErrorGradients(TMatrixT<Real_t> & dY,
                                         const TMatrixT<Real_t> &Y,
                                         const TMatrixT<Real_t> &output);

    /** Sigmoid transformation is implicitly applied, thus \p output should
     *  hold the linear activations of the last layer in the net. */
   static Real_t CrossEntropy(const TMatrixT<Real_t> &Y,
                              const TMatrixT<Real_t> &output);

   static void CrossEntropyGradients(TMatrixT<Real_t> & dY,
                                     const TMatrixT<Real_t> & Y,
                                     const TMatrixT<Real_t> & output);
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
   static void Sigmoid(TMatrixT<Real_t> &YHat,
                        const TMatrixT<Real_t> & );
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

   static Real_t L1Regularization(const TMatrixT<Real_t> & W);
   static void AddL1RegularizationGradients(TMatrixT<Real_t> & A,
                                            const TMatrixT<Real_t> & W,
                                            Real_t weightDecay);

   static Real_t L2Regularization(const TMatrixT<Real_t> & W);
   static void AddL2RegularizationGradients(TMatrixT<Real_t> & A,
                                            const TMatrixT<Real_t> & W,
                                            Real_t weightDecay);
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

   static void InitializeGauss(TMatrixT<Real_t> & A);

   static void InitializeUniform(TMatrixT<Real_t> & A);

   static void InitializeIdentity(TMatrixT<Real_t> & A);

   static void InitializeZero(TMatrixT<Real_t> & A);

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
   static void Dropout(TMatrixT<Real_t> & A, Real_t dropoutProbability);

   ///@}
};

} // namespace DNN
} // namespace TMVA

#endif
