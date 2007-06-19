// @(#)root/tmva $\Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MetricEuler                                                         *
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

#ifndef ROOT_TMVA_MetricEuler
#define ROOT_TMVA_MetricEuler

#include "Riostream.h"
#include <vector>
#include "TObject.h"
#include "TString.h"

#ifndef ROOT_TMVA_IMetric
#include "IMetric.h"
#endif


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MetricEuler                                                          //
//                                                                      //
// distance between two points in parameter space                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


namespace TMVA {

   class IMetric;

   class MetricEuler : public IMetric {

   public:

      MetricEuler();
      virtual ~MetricEuler() {}

      virtual Double_t Distance( std::vector<Double_t>& pointA, std::vector<Double_t>& pointB );

   private:

      ClassDef(MetricEuler,0) // calculates the "distance" between two points
   };

} // namespace TMVA

#endif


