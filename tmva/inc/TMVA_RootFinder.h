// @(#)root/tmva $Id: TMVA_RootFinder.h,v 1.1 2006/05/08 12:46:31 brun Exp $    
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_RootFinder                                                       *
 *                                                                                *
 * Description:                                                                   *
 *      Root finding using Brents algorithm                                       *
 *      (translated from CERNLIB function RZERO)                                  *
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
 *      MPI-KP Heidelberg, Germany                                                * 
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_RootFinder.h,v 1.1 2006/05/08 12:46:31 brun Exp $    
 **********************************************************************************/

#ifndef ROOT_TMVA_RootFinder
#define ROOT_TMVA_RootFinder

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_RootFinder                                                      //
//                                                                      //
// Root finding using Brents algorithm                                  //
// (translated from CERNLIB function RZERO)                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObject.h"

class TMVA_RootFinder : public TObject {

 public:

  TMVA_RootFinder( Double_t (*rootVal)( Double_t ),
		   Double_t rootMin, Double_t rootMax,
		   Int_t    maxIterations = 100, 
		   Double_t absTolerance  = 0.0 );
  virtual ~TMVA_RootFinder( void );

  // returns the root of the function
  Double_t Root( Double_t refValue );

 private:

  Double_t fRootMin;
  Double_t fRootMax;  
  Int_t    fMaxIter;
  Double_t fAbsTol; 

  // function pointer
  Double_t (*fGetRootVal)( Double_t );

  ClassDef(TMVA_RootFinder,0) //Root finding using Brents algorithm

};

#endif
