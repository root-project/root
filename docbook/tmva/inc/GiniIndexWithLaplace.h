// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GiniIndexWithLaplace                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the GiniIndex With Laplace correction           *
 *              as separation criterion                                           *
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
 * (http://ttmva.sourceforge.net/LICENSE)                                         *
 **********************************************************************************/

#ifndef ROOT_TMVA_GiniIndexWithLaplace
#define ROOT_TMVA_GiniIndexWithLaplace

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GiniIndexWithLaplace                                                 //
//                                                                      //
// Implementation of the GiniIndex With Laplace correction              // 
//     as separation criterion                                          //
//                                                                      //
//     Large Gini Indices (maximum 0.5) mean , that the sample is well  //
//     mixed (same amount of signal and bkg)                            //
//     bkg. Small Indices mean, well separated.                         //
//     general defniniton:                                              //     
//     Gini(Sample M) = 1 - (c(1)/N)^2 - (c(2)/N)^2 .... - (c(k)/N)^2   // 
//     Where: M is a smaple of whatever N elements (events)             //
//            that belong to K different classes                        //
//            c(k) is the number of elements that belong to class k     //
//     for just Signal and Background classes this boils down to:       //
//     the "Lapalace correction to the probability distribution would   //
//       turn the c(1)/N into (c(1)+1)/(N+2)                            //
//     using this the simple Gini Index  for two classes                //
//               Gini(Sample) = 2s*b/(s+b)^2                            //  
//       turns into                                                     //
//        GiniLaplace(Sample) = 2(s*b+s+b+1)/(s+b+2)^2                  //  
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA {

   class GiniIndexWithLaplace : public SeparationBase {
      
   public:
      
      // construtor for the GiniIndexWithLaplace
      GiniIndexWithLaplace() { fName="GiniLaplace"; }

      // copy constructor
      GiniIndexWithLaplace( const GiniIndexWithLaplace& g): SeparationBase(g) {}

      //destructor
      virtual ~GiniIndexWithLaplace(){}
      
      // Return the separation index (a measure for "purity" of the sample")
      virtual Double_t GetSeparationIndex( const Double_t &s, const Double_t &b );

   protected:
      
      ClassDef(GiniIndexWithLaplace,0) // Implementation of the GiniIndexWithLaplace as separation criterion
   };  

} // namespace TMVA

#endif

