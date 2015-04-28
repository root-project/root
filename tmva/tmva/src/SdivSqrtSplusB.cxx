// @(#)root/tmva $Id$   
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SdivSqrtSplusB                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the SdivSqrtSplusB as separation criterion      *
 *              s / sqrt( s+b )                                                   * 
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

#include "TMath.h"
#include "TMVA/SdivSqrtSplusB.h"

ClassImp(TMVA::SdivSqrtSplusB)

//_______________________________________________________________________
Double_t  TMVA::SdivSqrtSplusB::GetSeparationIndex( const Double_t &s, const Double_t &b )
{
   // Index = S/sqrt(S+B)  (statistical significance)                 
   if (s+b > 0) return s / TMath::Sqrt(s+b);
   else return 0;
}


 
//_______________________________________________________________________
Double_t TMVA::SdivSqrtSplusB::GetSeparationGain(const Double_t &nSelS, const Double_t& nSelB,
                                                 const Double_t& nTotS, const Double_t& nTotB)
{
   // Separation Gain:
   // the measure of how the quality of separation of the sample increases
   // by splitting the sample e.g. into a "left-node" and a "right-node"
   // (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
   // this is then the quality crition which is optimized for when trying
   // to increase the information in the system (making the best selection

   if ( (nTotS-nSelS)==nSelS && (nTotB-nSelB)==nSelB) return 0.;

   Double_t parentIndex = (nTotS+nTotB) *this->GetSeparationIndex(nTotS,nTotB);

   Double_t leftIndex   = ( ((nTotS - nSelS) + (nTotB - nSelB))
                            * this->GetSeparationIndex(nTotS-nSelS,nTotB-nSelB) );
   Double_t rightIndex  = (nSelS+nSelB) * this->GetSeparationIndex(nSelS,nSelB);

   //Double_t diff = parentIndex - leftIndex - rightIndex;
   Double_t diff = (parentIndex - leftIndex - rightIndex)/(nTotS+nTotB);

   if(diff<fPrecisionCut ) {
      // std::cout << " Warning value in GetSeparation is below numerical presicion " 
      //           << diff/parentIndex 
      //           << std::endl;
      return 0;
   }

   return diff;
}

