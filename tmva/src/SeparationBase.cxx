/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::SeparationBase                                                  *
 *                                                                                *
 * Description: An interface to different separation critiera useded in various   *
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
 *          this is then the quality crition which is optimized for when trying   *
 *          to increase the information in the system (making the best selection  *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Xavier Prudent  <prudent@lapp.in2p3.fr>  - LAPP, France                   *
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-KP Heidelberg, Germany     *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland,                                                        * 
 *      U. of Victoria, Canada,                                                   * 
 *      Heidelberg U., Germany,                                                   * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 **********************************************************************************/

#include "TMVA/SeparationBase.h"

ClassImp(TMVA::SeparationBase)

//_______________________________________________________________________
   Double_t TMVA::SeparationBase::GetSeparationGain(const Double_t &nSelS, const Double_t& nSelB, 
                                                    const Double_t& nTotS, const Double_t& nTotB)
{
   // Separation Gain:                                                     
   // the measure of how the quality of separation of the sample increases 
   // by splitting the sample e.g. into a "left-node" and a "right-node"   
   // (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right) 
   // this is then the quality crition which is optimized for when trying  
   // to increase the information in the system (making the best selection             

   Double_t parentIndex = (nTotS+nTotB) *this->GetSeparationIndex(nTotS,nTotB);
   Double_t leftIndex   = ( ((nTotS - nSelS) + (nTotB - nSelB))
                            * this->GetSeparationIndex(nTotS-nSelS,nTotB-nSelB) );
   Double_t rightIndex  = (nSelS+nSelB) * this->GetSeparationIndex(nSelS,nSelB);
    
   return (parentIndex - leftIndex - rightIndex)/(parentIndex);   
}


