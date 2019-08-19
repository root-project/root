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

namespace TMVA
{
namespace DNN
{ 
namespace CNN
{

struct TDescriptors {};
//______________________________________________________________________________
//
// Keeps the descriptors for the CNN 
//______________________________________________________________________________

template<typename Layer_t>
struct TCNNDescriptors : public TDescriptors {
   using LayerDescriptor_t   = typename Layer_t::LayerDescriptor_t;   // Main layer operation
   using HelperDescriptor_t  = typename Layer_t::HelperDescriptor_t;  // Used to define possible helpers for the layers (e.g. activations)
   using WeightsDescriptor_t = typename Layer_t::WeightsDescriptor_t; // The weights that are modified (e.g filters)

   LayerDescriptor_t   LayerDescriptor;
   HelperDescriptor_t  HelperDescriptor;
   WeightsDescriptor_t WeightsDescriptor;
   
   void InitializeDescriptors() {Layer_t::InitializeDescriptors();};
};

} // namespace CNN
} // namespace DNN
} // namespace TMVA

#endif
