// @(#)root/tmva $Id$
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
 *      Helge Voss      <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany      *
 *      Kai Voss        <Kai.Voss@cern.ch>       - U. of Victoria, Canada         *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      U. of Victoria, Canada                                                    * 
 *      Heidelberg U., Germany                                                    * 
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

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA {

   class MisClassificationError : public SeparationBase {

   public:

      // consturctor for the Misclassification error
      MisClassificationError() { fName = "MisCl"; }

      // copy constructor
      MisClassificationError( const MisClassificationError& g): SeparationBase(g) {}

      // destructor
      virtual ~MisClassificationError() {}

      // Return the separation index: 1-max(p,1-p)
      virtual Double_t  GetSeparationIndex( const Double_t &s, const Double_t &b );
   protected:

      ClassDef(MisClassificationError,0) // Implementation of the MisClassificationError as separation criterion
   };
 
} // namespace TMVA

#endif


