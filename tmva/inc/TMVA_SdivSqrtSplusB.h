// @(#)root/tmva $Id: TMVA_SdivSqrtSplusB.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_SdivSqrtSplusB                                                   *
 *                                                                                *
 * Description: Implementation of the SdivSqrtSplusB as separation criterion      *
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
 * $Id: TMVA_SdivSqrtSplusB.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $       
 **********************************************************************************/
#ifndef ROOT_TMVA_SdivSqrtSplusB
#define ROOT_TMVA_SdivSqrtSplusB

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_SdivSqrtSplusB                                                  //
//                                                                      //
// Implementation of the SdivSqrtSplusB as separation criterion         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA_SeparationBase.h"
#endif

class TMVA_SdivSqrtSplusB : public TMVA_SeparationBase {

 public:

  TMVA_SdivSqrtSplusB() { fName = "StatSig"; }
  virtual ~TMVA_SdivSqrtSplusB() {}

 protected:

  virtual Double_t  GetSeparationIndex( const Double_t &s, const Double_t &b );
 
 ClassDef(TMVA_SdivSqrtSplusB,0) //Implementation of the SdivSqrtSplusB as separation criterion
  
};

#endif

