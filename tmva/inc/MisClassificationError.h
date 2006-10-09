// @(#)root/tmva $Id: MisClassificationError.h,v 1.7 2006/08/30 22:19:59 andreas.hoecker Exp $
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MisClassificationError                                                *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MisClassificationError
#define ROOT_TMVA_MisClassificationError

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MisClassificationError                                               //
//                                                                      //
// Implementation of the MisClassificationError as separation criterion //
//                                                                      //
//       criterion: 1-max(p, 1-p)                                       //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA {

   class MisClassificationError : public SeparationBase {

   public:

      // consturctor for the Misclassification error
      MisClassificationError() { fName = "MisCl"; }
      // destructor
      virtual ~MisClassificationError() {}

   protected:

      // Return the separation index: 1-max(p,1-p)
      virtual Double_t  GetSeparationIndex( const Double_t &s, const Double_t &b );
 
      ClassDef(MisClassificationError,0) // Implementation of the MisClassificationError as separation criterion
  
         };
 
} // namespace TMVA

#endif


