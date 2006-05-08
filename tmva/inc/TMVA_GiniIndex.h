// @(#)root/tmva $Id: TMVA_GiniIndex.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $ 
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GiniIndex                                                        *
 *                                                                                *
 * Description: Implementation of the GiniIndex as separation criterion           *
 *              Large Gini Indices mean, that the sample is well mixed signal and *
 *              bkg. Small Indices mean, well separated. Hence we want a minimal  *
 *              Gini_left + Gini_right, or a                                      *
 *                MAXIMAl Gini_parent-Gini_left-Gini_right                        * 
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
 * (http://tmva.sourceforge.net/license.txt)                                      *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_GiniIndex.h,v 1.7 2006/05/02 23:27:40 helgevoss Exp $ 
 **********************************************************************************/
#ifndef ROOT_TMVA_GiniIndex
#define ROOT_TMVA_GiniIndex

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_GiniIndex                                                       //
//                                                                      //
// Implementation of the GiniIndex as separation criterion              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA_SeparationBase.h"
#endif

class TMVA_GiniIndex : public TMVA_SeparationBase {
  
 public:
  
  TMVA_GiniIndex() { fName="Gini"; }
  virtual ~TMVA_GiniIndex(){}
  
 protected:
  
  virtual Double_t GetSeparationIndex( const Double_t &s, const Double_t &b );
 
 ClassDef(TMVA_GiniIndex,0) //Implementation of the GiniIndex as separation criterion
  
};

#endif

