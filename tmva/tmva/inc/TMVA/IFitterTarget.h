// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : IFitterTarget                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Interface for generic fitter                                              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker <Andreas.Hocker@cern.ch> - CERN, Switzerland              *
 *      Peter Speckmayer <speckmay@mail.cern.ch>  - CERN, Switzerland             *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_IFitterTarget
#define ROOT_TMVA_IFitterTarget

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IFitterTarget                                                        //
//                                                                      //
// Interface for a fitter "target"                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include "TMVA/Types.h"


namespace TMVA {

   class IFitterTarget {

   public:

      IFitterTarget();

      virtual ~IFitterTarget() {}

      virtual Double_t EstimatorFunction( std::vector<Double_t>& parameters ) = 0;

      // function to notify the FitterTarget of the progress status of the fitter
      // sender : "GA", "MC", ...
      // progress : "init", "iteration", "last", "stop"
      virtual void     ProgressNotifier ( TString /*sender*/, TString /* progress */ ) {}

   private:

      ClassDef(IFitterTarget,0); // base class for a fitter "target"
   };

} // namespace TMVA

#endif
