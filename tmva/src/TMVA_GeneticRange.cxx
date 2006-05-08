// @(#)root/tmva $Id: TMVA_GeneticRange.cpp,v 1.4 2006/05/02 12:01:35 andreas.hoecker Exp $    
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
 * $Id: TMVA_GeneticRange.cpp,v 1.4 2006/05/02 12:01:35 andreas.hoecker Exp $
 **********************************************************************************/

#include "TMVA_GeneticRange.h"
#include <iostream>

//_______________________________________________________________________
//                                                                      
// Range definition for genetic algorithm                               
//                                                                      
//_______________________________________________________________________

TMVA_GeneticRange::TMVA_GeneticRange( TRandom *rnd, Double_t f, Double_t t )
{
  from = f;
  to = t;
  totalLength = t-f;

  randomGenerator = rnd;
}

Double_t TMVA_GeneticRange::random( Bool_t near, Double_t value, Double_t spread, Bool_t mirror )
{
  if( near ){
    Double_t ret;
    ret = randomGenerator->Gaus( value, totalLength*spread );
    if( mirror ) return reMapMirror( ret );
    else return reMap( ret );
  }
  return randomGenerator->Uniform(from, to);
}

Double_t TMVA_GeneticRange::reMap( Double_t val )
{
  if( from >= to ) return val;
  if( val <= from ) return reMap( (val-from) + to );
  if( val > to ) return reMap( (val-to) + from );
  return val;
}

Double_t TMVA_GeneticRange::reMapMirror( Double_t val )
{
  if( from >= to ) return val;
  if( val <= from ) return reMap( from - (val-from) );
  if( val > to ) return reMap( to - (val-to)  );
  return val;
}

TMVA_GeneticRange::~TMVA_GeneticRange(){}

