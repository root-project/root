/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Interval                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *          Extension of the Interval to "logarithmic" invarvals                  *
 *                                                                                *
 *                                                                                *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Helge Voss <helge.voss@cern.ch>  - MPI-K Heidelberg, Germany              *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
/* Begin_Html
<center><h2>the TMVA::Interval Class</h2></center>

<ul>
   <li> LogInterval definition, continuous and discrete
   <ul>
         <li>  LogInterval(min,max)  : a continous interval [min,max]
         <li>  LogInterval(min,max,n): a "discrete interval" [min,max], i.e the n numbers:<br>
         1,10,100,1000  <br>
         1,2,4,8,16,32,64,128,512,1024 <br>
         or alike .. <br>

   </ul>
</ul>
<pre>
    Example:
 LogInterval(1,10000,5)
     i=0 --> 1               note: StepSize(ibin=0) =  not defined !!
     i=1 --> 10                    StepSize(ibin=1) = 9
     i=2 --> 100                   StepSize(ibin=2) = 99
     i=3 --> 1000                  StepSize(ibin=3) = 999
     i=4 --> 10000                 StepSize(ibin=4) = 9999

 LogInterval(1,1000,11)
    i=0 --> 1
    i=1 --> 1.99526
    i=2 --> 3.98107
    i=3 --> 7.94328
    i=4 --> 15.8489
    i=5 --> 31.6228
    i=6 --> 63.0957
    i=7 --> 125.893
    i=8 --> 251.189
    i=9 --> 501.187
    i=10 --> 1000

 LogInterval(1,1024,11)
    i=0 --> 1
    i=1 --> 2
    i=2 --> 4
    i=3 --> 8
    i=4 --> 16
    i=5 --> 32
    i=6 --> 64
    i=7 --> 128
    i=8 --> 256
    i=9 --> 512
    i=10 --> 1024


</pre>
End_Html */

#include "TMath.h"
#include "TRandom3.h"
#include "ThreadLocalStorage.h"

#include "TMVA/LogInterval.h"
#include "TMVA/MsgLogger.h"

ClassImp(TMVA::LogInterval)

//_______________________________________________________________________
TMVA::LogInterval::LogInterval( Double_t min, Double_t max, Int_t nbins ) :
TMVA::Interval(min,max,nbins)
{
   if (min<=0) Log() << kFATAL << "logarithmic intervals have to have Min>0 !!" << Endl;
}

TMVA::LogInterval::LogInterval( const LogInterval& other ) :
   TMVA::Interval(other)
{
}

//_______________________________________________________________________
TMVA::LogInterval::~LogInterval()
{
   // destructor
}

//_______________________________________________________________________
Double_t TMVA::LogInterval::GetElement( Int_t bin ) const
{
   // calculates the value of the "number" bin in a discrete interval.
   // Parameters:
   //        Double_t position
   //
   if (fNbins <= 0) {
      Log() << kFATAL << "GetElement only defined for discrete value LogIntervals" << Endl;
      return 0.0;
   }
   else if (bin < 0 || bin >= fNbins) {
      Log() << kFATAL << "bin " << bin << " out of range: interval *bins* count from 0 to " << fNbins-1  << Endl;
      return 0.0;
   }
   return  TMath::Exp(TMath::Log(fMin)+((Double_t)bin) /((Double_t)(fNbins-1))*log(fMax/fMin));
}

//_______________________________________________________________________
Double_t TMVA::LogInterval::GetStepSize( Int_t iBin )  const
{
   // retuns the step size between the numbers of a "discrete LogInterval"
   if (fNbins <= 0) {
      Log() << kFATAL << "GetElement only defined for discrete value LogIntervals" << Endl;
   }
   if (iBin<0) {
      Log() << kFATAL << "You asked for iBin=" << iBin
            <<" in interval .. and.. sorry, I cannot let this happen.."<<Endl;
   }
   return (GetElement(TMath::Max(iBin,0))-GetElement(TMath::Max(iBin-1,0)));
}

//_______________________________________________________________________
Double_t TMVA::LogInterval::GetRndm( TRandom3& rnd )  const
{
   // get uniformely distributed number within interval
   return TMath::Exp(rnd.Rndm()*(TMath::Log(fMax/fMin) - TMath::Log(fMin)) + TMath::Log(fMin));
}

Double_t TMVA::LogInterval::GetWidth() const
{
   return fMax - fMin;
}
Double_t TMVA::LogInterval::GetMean()  const
{
   return (fMax + fMin)/2;
}

TMVA::MsgLogger& TMVA::LogInterval::Log() const {
  TTHREAD_TLS_DECL_ARG(MsgLogger,logger,"LogInterval");   // message logger
  return logger;
}
