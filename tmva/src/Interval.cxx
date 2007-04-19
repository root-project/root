// @(#)root/tmva $\Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::Interval                                                        *
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
 *      MPI-K Heidelberg, Germany ,                                               *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 *                                                                                *
 * File and Version Information:                                                  *
 **********************************************************************************/

//_______________________________________________________________________
//                                                                      
// Interval definition, continuous and discrete                          
//_______________________________________________________________________

#include "TMVA/Interval.h"
#include "TMath.h"

ClassImp(TMVA::Interval)

//_______________________________________________________________________
TMVA::Interval::Interval( Double_t min, Double_t max, Int_t nbins ) 
   : fLogger( "Interval" )
{
   // defines minimum and maximum of an interval
   // when nbins == 0, interval describes a discrete distribution (equally distributed in the interval)
   // when nbins > 0, interval describes a continous interval
   //
   fMin   = min;
   fMax   = max;
   fNbins = nbins;
   if( fMax - fMin == 0 ){
       fLogger << kFATAL << "maximum lower than minimum" << Endl;
   }
   if( nbins < 0 ){
       fLogger << kFATAL << "nbins < 0" << Endl;
       return;
   }
   if( nbins == 1 ){
       fLogger << kFATAL << "interval has to have at least 2 bins if discrete" << Endl;
       return;
   }
}

//_______________________________________________________________________
TMVA::Interval::~Interval()
{
   // destructor
}

//_______________________________________________________________________
Double_t TMVA::Interval::GetElement( Int_t bin ) const
{
   // calculates the value of the "number" bin in a discrete interval. 
   // Parameters:
   //        Double_t position 
   //
   if( fNbins <= 0 ){
       fLogger << kFATAL << "GetElement only possible for discrete values" << Endl;
       return 0.0;
   }
   if( bin < 0 || bin >= fNbins ){
       fLogger << kFATAL << "bin " << bin << " out of interval [0," << fNbins << ")" << Endl;
       return 0.0;
   }
   
   return fMin + ( (Double_t(bin)/(fNbins-1)) *(fMax - fMin) );
}


