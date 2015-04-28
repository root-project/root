/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : BDTEventWrapper                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *                                                                                *
 *                                                                                *
 * Author: Doug Schouten (dschoute@sfu.ca)                                        *
 *                                                                                *
 *                                                                                *
 * Copyright (c) 2007:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Texas at Austin, USA                                                *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_BDTEventWrapper
#include "TMVA/BDTEventWrapper.h"
#endif

using namespace TMVA;

BDTEventWrapper::BDTEventWrapper(const Event* e) : fEvent(e) {
   // constuctor

  fBkgWeight = 0.0;
  fSigWeight = 0.0;
}

BDTEventWrapper::~BDTEventWrapper() {
   // destructor
}

void BDTEventWrapper::SetCumulativeWeight(Bool_t type, Double_t weight) {
   // Set the accumulated weight, for sorted signal/background events
   /**
    * @param fType - true for signal, false for background
    * @param weight - the total weight
    */

   if(type) fSigWeight = weight;
   else fBkgWeight = weight;
}

Double_t BDTEventWrapper::GetCumulativeWeight(Bool_t type) const {
   // Get the accumulated weight

   if(type) return fSigWeight;
   return fBkgWeight;
}
