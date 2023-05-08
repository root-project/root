// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : GeneticRange                                                          *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *    Range definition for genetic algorithm                                      *
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

#ifndef ROOT_TMVA_GeneticRange
#define ROOT_TMVA_GeneticRange

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// GeneticRange                                                         //
//                                                                      //
// Range definition for genetic algorithm                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/Interval.h"

class TRandom3;

namespace TMVA {

   class GeneticRange {

   public:

      GeneticRange( TRandom3* rnd, Interval *interval );
      virtual ~GeneticRange();

      Double_t Random( Bool_t near = kFALSE, Double_t value=0, Double_t spread=0.1, Bool_t mirror=kFALSE );
      Double_t RandomDiscrete();

      Double_t GetFrom()        { return fFrom; }
      Double_t GetTo()          { return fTo; }
      Double_t GetTotalLength() { return fTotalLength; }

   private:

      Double_t fFrom, fTo;    ///< the constraints of the coefficient
      Int_t fNbins;

      Interval *fInterval;     ///< holds the complete information of the interval
      Double_t fTotalLength;  ///< the distance between the lower and upper constraints

      // maps the values thrown outside of the ]from,to] interval back to the interval
      // the values which leave the range on the from-side, are mapped in to the to-side
      Double_t ReMap( Double_t val );

      // same as before, but the values leaving the allowed range, are mirrored into the range.
      Double_t ReMapMirror( Double_t val );

      TRandom3* fRandomGenerator;  ///< the randomGenerator for calculating the new values

      ClassDef(GeneticRange,0); // Range definition for genetic algorithm
   };

} // namespace TMVA

#endif
