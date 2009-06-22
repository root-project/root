// @(#)root/tmva $Id$    
// Author: Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SVKernelFunction                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Kernel for Support Vector Machine                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <azemla@cern.ch>        - IFJ PAN, Krakow, Poland          *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *   
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SVKernelFunction
#define ROOT_TMVA_SVKernelFunction

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace TMVA {

	class SVEvent;
	class SVKernelFunction {

   public:

      SVKernelFunction();
      SVKernelFunction( Float_t );
      ~SVKernelFunction();

      Float_t Evaluate( SVEvent* ev1, SVEvent* ev2 );

   private:

      Float_t fGamma;   // documentation
	};
}

#endif
