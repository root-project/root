// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SeparationBase                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: An interface to different separation criteria used in various     *
 *              training algorithms, as there are:                                *
 *              Gini-Index, Cross Entropy, Misclassification Error, e.t.c.        *
 *                                                                                *
 *          There are two things: the Separation Index, and the Separation Gain   *
 *          Separation Index:                                                     *
 *          Measure of the "purity" of a sample. If all elements (events) in the  *
 *          sample belong to the same class (e.g. signal or backgr), than the     *
 *          separation index is 0 (meaning 100% purity (or 0% purity as it is     *
 *          symmetric. The index becomes maximal, for perfectly mixed samples     *
 *          eg. purity=50% , N_signal = N_bkg                                     *
 *                                                                                *
 *          Separation Gain:                                                      *
 *          the measure of how the quality of separation of the sample increases  *
 *          by splitting the sample e.g. into a "left-node" and a "right-node"    *
 *          (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)  *
 *          this is then the quality criterion which is optimized for when trying *
 *          to increase the information in the system (making the best selection  *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      Heidelberg U., Germany                                                    *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SeparationBase
#define ROOT_TMVA_SeparationBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SeparationBase                                                       //
//                                                                      //
// An interface to calculate the "SeparationGain" for different         //
// separation criteria used in various training algorithms              //
//                                                                      //
// There are two things: the Separation Index, and the Separation Gain  //
// Separation Index:                                                    //
// Measure of the "purity" of a sample. If all elements (events) in the //
// sample belong to the same class (e.g. signal or background), than the//
// separation index is 0 (meaning 100% purity (or 0% purity as it is    //
// symmetric. The index becomes maximal, for perfectly mixed samples    //
// eg. purity=50% , N_signal = N_bkg                                    //
//                                                                      //
// Separation Gain:                                                     //
// the measure of how the quality of separation of the sample increases //
// by splitting the sample e.g. into a "left-node" and a "right-node"   //
// (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right) //
// this is then the quality criterion which is optimized for when trying//
// to increase the information in the system (making the best selection //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

#include "TString.h"

#include "TMath.h"

#include <limits>

namespace TMVA {

   class SeparationBase {

   public:

      // default constructor
      SeparationBase();

      //copy constructor
      SeparationBase( const SeparationBase& s );

      // destructor
      virtual ~SeparationBase(){}

      // Return the gain in separation of the original sample is split in two sub-samples
      // (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
      virtual Double_t GetSeparationGain( const Double_t nSelS, const Double_t nSelB,
                                          const Double_t nTotS, const Double_t nTotB );

      // Return the separation index (a measure for "purity" of the sample")
      virtual Double_t GetSeparationIndex( const Double_t s, const Double_t b ) = 0;

      // Return the name of the concrete Index implementation
      const TString& GetName() { return fName; }

   protected:

      TString fName;  // name of the concrete Separation Index implementation

      Double_t fPrecisionCut;

      ClassDef(SeparationBase,0); // Interface to different separation criteria used in training algorithms
   };


} // namespace TMVA

#endif
