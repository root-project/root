// @(#)root/tmva $Id$    
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
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
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

#include "TMath.h"
#include "TRandom3.h"

#include "TMVA/Interval.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::Interval)

TMVA::MsgLogger* TMVA::Interval::fgLogger = 0;

//_______________________________________________________________________
TMVA::Interval::Interval( Double_t min, Double_t max, Int_t nbins ) : 
   fMin(min),
   fMax(max),
   fNbins(nbins)
{
   if (!fgLogger) fgLogger = new MsgLogger("Interval");

   // defines minimum and maximum of an interval
   // when nbins == 0, interval describes a discrete distribution (equally distributed in the interval)
   // when nbins > 0, interval describes a continous interval
   //
   if (fMax - fMin < 0) Log() << kFATAL << "maximum lower than minimum" << Endl;
   if (nbins < 0) {
      Log() << kFATAL << "nbins < 0" << Endl;
      return;
   }
   else if (nbins == 1) {
      Log() << kFATAL << "interval has to have at least 2 bins if discrete" << Endl;
      return;
   }
}

TMVA::Interval::Interval( const Interval& other ) :
   fMin  ( other.fMin ),
   fMax  ( other.fMin ),
   fNbins( other.fNbins )
{
   if (!fgLogger) fgLogger = new MsgLogger("Interval");
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
   if (fNbins <= 0) {
      Log() << kFATAL << "GetElement only possible for discrete values" << Endl;
      return 0.0;
   }
   else if (bin < 0 || bin >= fNbins) {
      Log() << kFATAL << "bin " << bin << " out of interval [0," << fNbins << ")" << Endl;
      return 0.0;
   }
   
   return fMin + ( (Double_t(bin)/(fNbins-1)) *(fMax - fMin) );
}

//_______________________________________________________________________
Double_t TMVA::Interval::GetRndm( TRandom3& rnd )  const
{
   // get uniformely distributed number within interval
   return rnd.Rndm()*(fMax - fMin) + fMin;
}


