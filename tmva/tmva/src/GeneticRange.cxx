// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::GeneticRange                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
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
 *                                                                                *
 * File and Version Information:                                                  *
 **********************************************************************************/

/*! \class TMVA::GeneticRange
\ingroup TMVA

Range definition for genetic algorithm.

*/

#include "TRandom3.h"

#include "TMVA/GeneticRange.h"
#include "TMVA/Interval.h"

ClassImp(TMVA::GeneticRange);

////////////////////////////////////////////////////////////////////////////////
/// defines the "f" (from) and "t" (to) of the coefficient
/// and takes a randomgenerator

TMVA::GeneticRange::GeneticRange( TRandom3*rnd, Interval *interval )
{
   fInterval = interval;

   fFrom = fInterval->GetMin();
   fTo   = fInterval->GetMax();
   fNbins= fInterval->GetNbins();
   fTotalLength = fTo-fFrom;

   fRandomGenerator = rnd;
}

////////////////////////////////////////////////////////////////////////////////
/// creates a new random value for the coefficient; returns a discrete value

Double_t TMVA::GeneticRange::RandomDiscrete()
{
   Double_t value = fRandomGenerator->Uniform(0, 1);
   return fInterval->GetElement( Int_t(value*fNbins) );
}

////////////////////////////////////////////////////////////////////////////////
/// creates a new random value for the coefficient
/// Parameters:
///     -  Bool_t near   : takes a random value near the current value
///     -  double value  : this is the current value
///     -  double spread : the sigma of the gaussian which is taken to calculate the new value
///     -  Bool_t mirror : if the new value would be outside of the range, mirror = false
///                        maps the value between the constraints by periodic boundary conditions.
///                        With mirror = true, the value gets "reflected" on the boundaries.

Double_t TMVA::GeneticRange::Random( Bool_t near, Double_t value, Double_t spread, Bool_t mirror )
{
   if (fInterval->GetNbins() > 0) {   // discrete interval
      return RandomDiscrete();
   }
   else if (fFrom == fTo) {
      return fFrom;
   }
   else if (near) {
      Double_t ret;
      ret = fRandomGenerator->Gaus( value, fTotalLength*spread );
      if (mirror ) return ReMapMirror( ret );
      else return ReMap( ret );
   }
   return fRandomGenerator->Uniform(fFrom, fTo);
}

////////////////////////////////////////////////////////////////////////////////
/// remapping the value to the allowed space

Double_t TMVA::GeneticRange::ReMap( Double_t val )
{
   if (fFrom >= fTo ) return val;
   if (val < fFrom ) return ReMap( (val-fFrom) + fTo );
   if (val >= fTo )    return ReMap( (val-fTo) + fFrom );
   return val;
}

////////////////////////////////////////////////////////////////////////////////
/// remapping the value to the allowed space by reflecting on the boundaries

Double_t TMVA::GeneticRange::ReMapMirror( Double_t val )
{
   if (fFrom >= fTo ) return val;
   if (val < fFrom  ) return ReMap( fFrom - (val-fFrom) );
   if (val >= fTo   ) return ReMap( fTo - (val-fTo)  );
   return val;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::GeneticRange::~GeneticRange()
{
}

