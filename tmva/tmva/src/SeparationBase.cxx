// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SeparationBase                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: An interface to different separation criteria used in various     *
 *              training algorithms, as there are:                                *
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

/*! \class TMVA::SeparationBase
\ingroup TMVA
An interface to calculate the "SeparationGain" for different
separation criteria used in various training algorithms

There are two things: the Separation Index, and the Separation Gain
Separation Index:
Measure of the "purity" of a sample. If all elements (events) in the
sample belong to the same class (e.g. signal or background), than the
separation index is 0 (meaning 100% purity (or 0% purity as it is
symmetric. The index becomes maximal, for perfectly mixed samples
eg. purity=50% , N_signal = N_bkg

Separation Gain:
the measure of how the quality of separation of the sample increases
by splitting the sample e.g. into a "left-node" and a "right-node"
(N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
this is then the quality criterion which is optimized for when trying
to increase the information in the system (making the best selection
*/
#include "TMVA/SeparationBase.h"

#include "TMath.h"
#include "TString.h"

#include <iostream>
#include <limits>

ClassImp(TMVA::SeparationBase);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TMVA::SeparationBase::SeparationBase() :
fName(""),
   fPrecisionCut(TMath::Sqrt(std::numeric_limits<double>::epsilon()))
{
   // default constructor
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TMVA::SeparationBase::SeparationBase( const SeparationBase& s ) :
   fName(s.fName),
   fPrecisionCut(TMath::Sqrt(std::numeric_limits<double>::epsilon()))
{
   // copy constructor
}

////////////////////////////////////////////////////////////////////////////////
/// Separation Gain:
/// the measure of how the quality of separation of the sample increases
/// by splitting the sample e.g. into a "left-node" and a "right-node"
/// (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
/// this is then the quality criterion which is optimized for when trying
/// to increase the information in the system (making the best selection

Double_t TMVA::SeparationBase::GetSeparationGain(const Double_t nSelS, const Double_t nSelB,
                                                 const Double_t nTotS, const Double_t nTotB)
{
   if ( (nTotS-nSelS)==nSelS && (nTotB-nSelB)==nSelB) return 0.;

   // Double_t parentIndex = (nTotS+nTotB) *this->GetSeparationIndex(nTotS,nTotB);

   // Double_t leftIndex   = ( ((nTotS - nSelS) + (nTotB - nSelB))
   //                          * this->GetSeparationIndex(nTotS-nSelS,nTotB-nSelB) );
   // Double_t rightIndex  = (nSelS+nSelB) * this->GetSeparationIndex(nSelS,nSelB);


   Double_t parentIndex = this->GetSeparationIndex(nTotS,nTotB);

   Double_t leftIndex   = ( ((nTotS - nSelS) + (nTotB - nSelB))/(nTotS+nTotB)
                            * this->GetSeparationIndex(nTotS-nSelS,nTotB-nSelB) );
   Double_t rightIndex  = (nSelS+nSelB)/(nTotS+nTotB) * this->GetSeparationIndex(nSelS,nSelB);

   Double_t diff = parentIndex - leftIndex - rightIndex;
   //Double_t diff = (parentIndex - leftIndex - rightIndex)/(nTotS+nTotB);

   if(diff<fPrecisionCut ) {
      // std::cout << " Warning value in GetSeparation is below numerical precision "
      //           << diff/parentIndex
      //           << std::endl;
      return 0;
   }

   return diff;
}


