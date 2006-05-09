// @(#)root/tmva $Id: TMVA_GeneticRange.cxx,v 1.2 2006/05/08 20:56:17 brun Exp $    
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA_GeneticRange                                                     *
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
 * (http://mva.sourceforge.net/license.txt)                                       *
 *                                                                                *
 * File and Version Information:                                                  *
 * $Id: TMVA_GeneticRange.cxx,v 1.2 2006/05/08 20:56:17 brun Exp $
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Range definition for genetic algorithm                               
//                                                                      
//_______________________________________________________________________

#include "TMVA_GeneticRange.h"
#include <iostream>

ClassImp(TMVA_GeneticRange)

//_______________________________________________________________________
TMVA_GeneticRange::TMVA_GeneticRange( TRandom *rnd, Double_t f, Double_t t )
{
  fFrom = f;
  fTo = t;
  fTotalLength = t-f;

  fRandomGenerator = rnd;
}

//_______________________________________________________________________
Double_t TMVA_GeneticRange::Random( Bool_t near, Double_t value, Double_t spread, Bool_t mirror )
{
  if( near ){
    Double_t ret;
    ret = fRandomGenerator->Gaus( value, fTotalLength*spread );
    if( mirror ) return ReMapMirror( ret );
    else return ReMap( ret );
  }
  return fRandomGenerator->Uniform(fFrom, fTo);
}

//_______________________________________________________________________
Double_t TMVA_GeneticRange::ReMap( Double_t val )
{
  if( fFrom >= fTo ) return val;
  if( val <= fFrom ) return ReMap( (val-fFrom) + fTo );
  if( val > fTo )    return ReMap( (val-fTo) + fFrom );
  return val;
}

//_______________________________________________________________________
Double_t TMVA_GeneticRange::ReMapMirror( Double_t val )
{
  if( fFrom >= fTo ) return val;
  if( val <= fFrom ) return ReMap( fFrom - (val-fFrom) );
  if( val > fTo )    return ReMap( fTo - (val-fTo)  );
  return val;
}

//_______________________________________________________________________
TMVA_GeneticRange::~TMVA_GeneticRange(){}

