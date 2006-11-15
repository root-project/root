// @(#)root/tmva $Id: GeneticRange.cxx,v 1.11 2006/10/10 17:43:51 andreas.hoecker Exp $    
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
 *      CERN, Switzerland,                                                        *
 *      U. of Victoria, Canada,                                                   *
 *      MPI-KP Heidelberg, Germany,                                               *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: GeneticRange.cxx,v 1.11 2006/10/10 17:43:51 andreas.hoecker Exp $
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Range definition for genetic algorithm                               
//                                                                      
//_______________________________________________________________________

#include "TMVA/GeneticRange.h"
#include "Riostream.h"

ClassImp(TMVA::GeneticRange)
   ;

//_______________________________________________________________________
TMVA::GeneticRange::GeneticRange( TRandom *rnd, Double_t f, Double_t t )
{
   // defines the "f" (from) and "t" (to) of the coefficient
   // and takes a randomgenerator
   //
   fFrom = f;
   fTo   = t;
   fTotalLength = t-f;

   fRandomGenerator = rnd;
}

//_______________________________________________________________________
Double_t TMVA::GeneticRange::Random( Bool_t near, Double_t value, Double_t spread, Bool_t mirror )
{
   // creates a new random value for the coefficient
   // Parameters:
   //        bool near : takes a random value near the current value
   //        double value : this is the current value
   //        double spread : the sigma of the gaussian which is taken to calculate the new value
   //        bool mirror : if the new value would be outside of the range, mirror = false
   //               maps the value between the constraints by periodic boundary conditions.
   //               With mirror = true, the value gets "reflected" on the boundaries.
   //
   if (near ){
      Double_t ret;
      ret = fRandomGenerator->Gaus( value, fTotalLength*spread );
      if (mirror ) return ReMapMirror( ret );
      else return ReMap( ret );
   }
   return fRandomGenerator->Uniform(fFrom, fTo);
}

//_______________________________________________________________________
Double_t TMVA::GeneticRange::ReMap( Double_t val )
{
   // remapping the value to the allowed space
   //
   if (fFrom >= fTo ) return val;
   if (val <= fFrom ) return ReMap( (val-fFrom) + fTo );
   if (val > fTo )    return ReMap( (val-fTo) + fFrom );
   return val;
}

//_______________________________________________________________________
Double_t TMVA::GeneticRange::ReMapMirror( Double_t val )
{
   // remapping the value to the allowed space by reflecting on the 
   // boundaries
   if (fFrom >= fTo ) return val;
   if (val <= fFrom ) return ReMap( fFrom - (val-fFrom) );
   if (val > fTo )    return ReMap( fTo - (val-fTo)  );
   return val;
}

//_______________________________________________________________________
TMVA::GeneticRange::~GeneticRange()
{
   // destructor
}

