// @(#)root/tmva $Id: TMVA_MethodANNBase.h,v 1.1 2006/05/08 12:46:31 brun Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MethodANNBase                                                    *
 *                                                                                *
 * Description:                                                                   *
 *      Base class for all MVA methods based on artificial neural networks (ANN)  *
 *      contains common functionality                                             *
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
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_MethodANNBase.h,v 1.1 2006/05/08 12:46:31 brun Exp $
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodANNBase
#define ROOT_TMVA_MethodANNBase

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MethodANNBase                                                   //
//                                                                      //
// Base class for all MVA methods using artificial neural networks      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include <vector>

class TMVA_MethodANNBase {

 public:

  TMVA_MethodANNBase( void );
  virtual ~TMVA_MethodANNBase() {}

 protected:

  // option string parser
  // first input in vector is number of cycles; additional inputs give
  // number of nodes for each layer (as many layers as inputs in vector)
  std::vector<Int_t>* parseOptionString( TString, Int_t, std::vector<Int_t>* );
 
 ClassDef(TMVA_MethodANNBase,0) //Base class for all MVA methods using artificial neural networks

};

#endif
