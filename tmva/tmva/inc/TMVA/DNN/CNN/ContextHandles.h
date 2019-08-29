// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/06/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Contains function enums for activation and output functions, as //
// well as generic evaluation functions, that delegate the call to //
// the corresponding evaluation kernel.                            //
/////////////////////////////////////////////////////////////////////

#ifndef TMVA_DNN_CNN_DESCRIPTORS
#define TMVA_DNN_CNN_DESCRIPTORS

#include <stddef.h>

namespace TMVA
{
namespace DNN
{ 
   struct TDescriptors {};
   struct TWorkspace {};
namespace CNN
{

//______________________________________________________________________________
//
// Keeps the descriptors for the CNN 
//______________________________________________________________________________

template<typename Layer_t>
struct TCNNDescriptors : public TMVA::DNN::TDescriptors {
   using LayerDescriptor_t   = typename Layer_t::LayerDescriptor_t;   // Main layer operation
   using HelperDescriptor_t  = typename Layer_t::HelperDescriptor_t;  // Used to define possible helpers for the layers (e.g. activations)
   using WeightsDescriptor_t = typename Layer_t::WeightsDescriptor_t; // The weights that are modified (e.g filters)

   LayerDescriptor_t   LayerDescriptor;
   HelperDescriptor_t  HelperDescriptor;
   WeightsDescriptor_t WeightsDescriptor;
};

template<typename Layer_t>
struct TCNNWorkspace : public TMVA::DNN::TWorkspace {
   using AlgorithmForward_t  = typename Layer_t::AlgorithmForward_t;  // Forward layer operation
   using AlgorithmBackward_t = typename Layer_t::AlgorithmBackward_t; // Backward layer operation
   using AlgorithmHelper_t   = typename Layer_t::AlgorithmHelper_t;   // Used for weight grad backward pass

   // FIXME: Add other cudnn types (algorithm preference etc.)
   using AlgorithmDataType_t = typename Layer_t::AlgorithmDataType_t;

   AlgorithmForward_t  AlgorithmForward;
   AlgorithmBackward_t AlgorithmBackward;
   AlgorithmHelper_t   HelperAlgorithm;

   AlgorithmDataType_t DataType;

   size_t * ForwardWorkspace;
   size_t * BackwardWorkspace;
   size_t * HelperWorkspace;

   size_t ForwardWorkspaceSize;
   size_t BackwardWorkspaceSize;
   size_t HelperWorkspaceSize;
};

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
