// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : RegressionVariance                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Calculate the separation criteria used in regression              *
 *                                                                                *
 *          There are two things: the Separation Index, and the Separation Gain   *
 *          Separation Index:                                                     *
 *          Measure of the "Variance" of a sample.                                *
 *                                                                                *
 *          Separation Gain:                                                      *
 *          the measure of how the quality of separation of the sample increases  *
 *          by splitting the sample e.g. into a "left-node" and a "right-node"    *
 *          (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)  *
 *          this is then the quality criteria which is optimized for when trying  *
 *          to increase the information in the system (making the best selection  *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
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

#ifndef ROOT_TMVA_RegressionVariance
#define ROOT_TMVA_RegressionVariance

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// RegressionVariance                                                   //
//                                                                      //
// Calculate the "SeparationGain" for Regression analysis               //
// separation criteria used in various training algorithms              //
//                                                                      //
// There are two things: the Separation Index, and the Separation Gain  //
// Separation Index:                                                    //
// Measure of the "Variance" of a sample.                               //
//                                                                      //
// Separation Gain:                                                     //
// the measure of how the quality of separation of the sample increases //
// by splitting the sample e.g. into a "left-node" and a "right-node"   //
// (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right) //
// this is then the quality criteria which is optimized for when trying //
// to increase the information in the system (making the best selection //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

#include "TString.h"

namespace TMVA {

   class RegressionVariance {

   public:

      //default constructor
      RegressionVariance(){fName = "Variance for Regression";}

      //copy constructor
   RegressionVariance( const RegressionVariance& s ): fName ( s.fName ) {}

      // destructor
      virtual ~RegressionVariance(){}

      // Return the gain in separation of the original sample is split in two sub-samples
      // (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
      Double_t GetSeparationGain( const Double_t nLeft, const Double_t targetLeft, const Double_t target2Left,
                                  const Double_t nTot, const Double_t targetTot, const Double_t target2Tot );

      // Return the separation index (a measure for "purity" of the sample")
      virtual Double_t GetSeparationIndex( const Double_t n, const Double_t target, const Double_t target2 );

      // Return the name of the concrete Index implementation
      TString GetName() { return fName; }

   protected:

      TString fName;  ///< name of the concrete Separation Index implementation

      ClassDef(RegressionVariance,0); // Interface to different separation criteria used in training algorithms
   };


} // namespace TMVA

#endif
