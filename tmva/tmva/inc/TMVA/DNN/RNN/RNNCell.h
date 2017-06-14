// @(#)root/tmva/tmva/dnn/rnn:$Id$
// Author: Saurav Shekhar 14/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RNNCell                                                               *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Saurav Shekhar    <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 * All rights reserved.                                                  *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 **********************************************************************************/

//#pragma once

//////////////////////////////////////////////////////////////////////
// <Description> //
//////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_RNN_CELL
#define TMVA_DNN_RNN_CELL

#include <cmath>
#include <iostream>

#include "TMatrix.h"
#include "TMVA/DNN/Functions.h"

namespace TMVA
{
namespace DNN
{
namespace RNN
{

//______________________________________________________________________________
//
//  The cell class
//______________________________________________________________________________

/** \class RNNCell
    Generic implementation
*/
template<typename Architecture_t>
	class TRNNCell
{
  
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

private:
  
  size_t fBatchSize;              ///< Batch size used for training and evaluation.

  size_t fStateSize;              ///< Hidden state size
  size_t fOutputSize;             ///< Output size 

  EActivationFunction fF;         ///< Activation function of the layer.

  Matrix_t *fOutput;
  Matrix_t *fGradients;

public:

  /** Constructor */
  TRNNCell(size_t BatchSize, 
           size_t StateSize,
           size_t OutputSize);

  /** Copy Constructor */
  TRNNCell(const TRNNCell &);

  /** Virtual destructor */
  virtual ~TRNNCell();

  /** Step function */
  virtual void step() = 0;

  /*! Initialize the weights according to the given initialization
   **  method. */
  virtual void Initialize(DNN::EInitialization m) = 0;

  /*! Computes activation of the layer for the given input. The input
  * must be in tensor form with the different matrices corresponding to
  * different events in the batch. Computes activations as well as
  * the first partial derivative of the activation function at those
  * activations. */
  virtual void Forward(Matrix_t *input) = 0;

  /*! Compute weight, bias and activation gradients. Uses the precomputed
  *  first partial derviatives of the activation function computed during
  *  forward propagation and modifies them. Must only be called directly
  *  at the corresponding call to Forward(...). */
  virtual void Backward(Matrix_t *gradients_backward,
                        const Matrix_t *activations_backward,
                        DNN::ERegularization r,
                        Scalar_t weightDecay) = 0;

  /** Prints the info about the layer */
  virtual void Print() const = 0;

};

//______________________________________________________________________________
//
//  Basic RNN Cell
//______________________________________________________________________________

/** \class BasicRNNCell
    Generic implementation
*/
template<typename Architecture_t>
	class TBasicRNNCell : public TRNNCell<Architecture_t>
{
  
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;

private:
  
  Matrix_t *U, *V, *W;  

public:

  /** Constructor */
  TBasicRNNCell(size_t BatchSize, 
           size_t StateSize,
           size_t OutputSize);

  /** Copy Constructor */
  TBasicRNNCell(const TRNNCell &);

  /** Virtual destructor */
  virtual ~TBasicRNNCell();

  /** Step function */
  virtual void step() = 0;

};


} // namespace RNN
} // namespace DNN
} // namespace TMVA

#endif
