// @(#)root/tmva $Id: TMVA_SeparationBase.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_SeparationBase                                                   *
 *                                                                                *
 * Description: An interface to different separation critiera useded in various   *
 *              training algorithms, as there are:                                *
 *              Gini-Index, Cross Entropy, Misclassification Error, e.t.c.        *
 *              usage: you have your starting event  sample with n=s+b  events    *
 *                     then you do some "selection" and split up this event sample*
 *                     with n1 = s1+b1, respective n2=s2+b2  events in it         *
 *                     obviously: s1+s2 = s, hence s2=s-s1, e.t.c.                *
 *                     the separation then is a measure of how you increase the   *
 *                     "quality" of your sample, by comparing                     *
 *                                                                                *
 *              The actual implementation should be done such, that the Index is  *
 *              maximal for maximally mixed samples (purity=50%, s=b)             *
 *               this means, that we are then actually looking for nodes with a   *
 *               MINIMAL separation Index.                                        *
 *               in this case then, LARGE GetSeparationGain                          *
 *                 (Index_parent - Index_left - Index_right) is what is optimised *
 *                 for                                                            *
 *                                                                                *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_SeparationBase.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $   
 **********************************************************************************/

#ifndef ROOT_TMVA_SeparationBase
#define ROOT_TMVA_SeparationBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_SeparationBase                                                  //
//                                                                      //
// An interface to different separation critiera used in various        //
// training algorithms                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"
#include "TString.h"

class TMVA_SeparationBase {

 public:

  TMVA_SeparationBase(){}
  virtual ~TMVA_SeparationBase(){};

  Double_t GetSeparationGain( const Double_t& nSelS, const Double_t& nSelB, 
			   const Double_t& nTotS, const Double_t& nTotB );

  virtual Double_t GetSeparationIndex( const Double_t &s, const Double_t &b ) = 0;

  TString GetName(){return fName;};

 protected:

  TString fName;
 
 ClassDef(TMVA_SeparationBase,0) //interface to different separation critiera used in  training algorithms
};


#endif
