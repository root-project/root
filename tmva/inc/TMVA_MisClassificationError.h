// @(#)root/tmva $Id: TMVA_MisClassificationError.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $
// Author: Andreas Hoecker, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_MisClassificationError                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation of the MisClassificationError as separation                *
 *      criterion: 1-max(p, 1-p)                                                  *
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
 * $Id: TMVA_MisClassificationError.h,v 1.6 2006/05/02 23:27:40 helgevoss Exp $  
 **********************************************************************************/

#ifndef ROOT_TMVA_MisClassificationError
#define ROOT_TMVA_MisClassificationError

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMVA_MisClassificationError                                          //
//                                                                      //
// Implementation of the MisClassificationError as separation criterion //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_
#include "TMVA_SeparationBase.h"
#endif

class TMVA_MisClassificationError : public TMVA_SeparationBase {

 public:

  TMVA_MisClassificationError() { fName = "MisCl"; }
  virtual ~TMVA_MisClassificationError() {}

 protected:

  virtual Double_t  GetSeparationIndex( const Double_t &s, const Double_t &b );
 
 ClassDef(TMVA_MisClassificationError,0) //Implementation of the MisClassificationError as separation criterion
  
};
 
#endif


