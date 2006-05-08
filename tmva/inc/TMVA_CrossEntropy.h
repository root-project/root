// @(#)root/tmva $Id: TMVA_CrossEntropy.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $       
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_CrossEntropy                                                     *
 *                                                                                *
 * Description: Implementation of the CrossEntropy as separation criterion        *
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
 * $Id: TMVA_CrossEntropy.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $       
 **********************************************************************************/

#ifndef ROOT_TMVA_CrossEntropy
#define ROOT_TMVA_CrossEntropy

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_CrossEntropy                                                    //
//                                                                      //
// Implementation of the CrossEntropy as separation criterion           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA_SeparationBase.h"
#endif

class TMVA_CrossEntropy : public TMVA_SeparationBase {

 public:

  TMVA_CrossEntropy() { fName = "CE"; }
  virtual ~TMVA_CrossEntropy(){}

 protected:

  virtual Double_t GetSeparationIndex( const Double_t &s, const Double_t &b );
 
 ClassDef(TMVA_CrossEntropy,0) //Implementation of the CrossEntropy as separation criterion
  
};

#endif

