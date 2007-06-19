// @(#)root/tmva $\Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : IMetric                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *       Fitter using a Genetic Algorithm                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
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

#ifndef ROOT_TMVA_IMetric
#define ROOT_TMVA_IMetric

#include "Riostream.h"
#include <vector>
#include "TObject.h"
#include "TString.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IMetric                                                               //
//                                                                      //
// distance between two points in parameter space                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


namespace TMVA {


   class IMetric {

   public:

      IMetric();
      virtual ~IMetric() {}

      virtual Double_t Distance( std::vector<Double_t>& pointA, std::vector<Double_t>& pointB ) = 0;
      void SetParameters( std::vector<Double_t>* parameters ) { fParameters = parameters; };
      std::vector<Double_t>* GetParameters() { return fParameters; };

   protected:
      std::vector<Double_t>* fParameters;
      
   private:
      
      ClassDef(IMetric,0) // calculates the "distance" between two points
   };

} // namespace TMVA

#endif


