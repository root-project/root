// @(#)root/tmva $Id$ 
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SdivSqrtSplusB                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description: Implementation of the SdivSqrtSplusB as separation criterion      *
 *              S/sqrt(S + B)                                                     *
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

#ifndef ROOT_TMVA_SdivSqrtSplusB
#define ROOT_TMVA_SdivSqrtSplusB

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// SdivSqrtSplusB                                                       //
//                                                                      //
// Implementation of the SdivSqrtSplusB as separation criterion         //
//   Index = S/sqrt(S+B)  (statistical significance)                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TMVA_SeparationBase
#include "TMVA/SeparationBase.h"
#endif

namespace TMVA {

   class SdivSqrtSplusB : public SeparationBase {

   public:

      //constructor for the "statistical significance" index
      SdivSqrtSplusB(): SeparationBase()  { fName = "StatSig"; }

      // copy constructor
      SdivSqrtSplusB( const SdivSqrtSplusB& g): SeparationBase(g) {}

      //destructor
      virtual ~SdivSqrtSplusB() {}

      // Return the gain in separation of the original sample is splitted in two sub-samples
      // (N * Index_parent) - (N_left * Index_left) - (N_right * Index_right)
      virtual Double_t GetSeparationGain( const Double_t& nSelS, const Double_t& nSelB,
                                  const Double_t& nTotS, const Double_t& nTotB );
      // return the Index (S/sqrt(S+B))
      virtual Double_t  GetSeparationIndex( const Double_t &s, const Double_t &b );

   protected:

      ClassDef(SdivSqrtSplusB,0) // Implementation of the SdivSqrtSplusB as separation criterion
   };

} // namespace TMVA

#endif

