/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_SeparationBase                                                   *
 *                                                                                *
 * Description: An interface to different separation critiera useded in various   *
 *              training algorithms, as there are:                                *
 *              Gini-Index, Cross Entropy, Misclassification Error, e.t.c.        *
 *              usage: you have your starting event  sample with n=s+b  events    *
 *                     then you do some "selection" and split up this event       *
 *                     sample with n1 = s1+b1, respective n2=s2+b2  events        *
 *                     in it obviously: s1+s2 = s, hence s2=s-s1, e.t.c.          *
 *                     the separation then is a measure of how you increase the   *
 *                     "quality" of your sample, by comparing                     *
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
 * (http://tmva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 **********************************************************************************/

#include "TMVA_SeparationBase.h"

ClassImp(TMVA_SeparationBase)

//_______________________________________________________________________
Double_t TMVA_SeparationBase::GetSeparationGain(const Double_t &nSelS, const Double_t& nSelB, 
					     const Double_t& nTotS, const Double_t& nTotB)
{
  Double_t parentIndex =  (nTotS+nTotB) *this->GetSeparationIndex(nTotS,nTotB);
  Double_t leftIndex   = ((nTotS - nSelS) + (nTotB - nSelB))
    * this->GetSeparationIndex(nTotS-nSelS,nTotB-nSelB);
  Double_t rightIndex  = (nSelS+nSelB) * this->GetSeparationIndex(nSelS,nSelB);
    
  //  return (parentIndex - leftIndex - rightIndex)/(nTotS+nTotB);   
  return (parentIndex - leftIndex - rightIndex)/(parentIndex);   
  //  return leftIndex + rightIndex;    
}


