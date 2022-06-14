/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Interval                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *          Extension of the Interval to "logarithmic" intervals                  *
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

#ifndef ROOT_TMVA_LogInterval
#define ROOT_TMVA_LogInterval

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Interval with non-equi distant bins                                      //
//      that are equi-distant in a logarithmic scale)                       //
//                                                                          //
// Interval definition, continuous and discrete                             //
//                                                                          //
//  Note: **bin** counting starts from ZERO unlike in ROOT histograms       //
//                                                                          //
//       ----------------                                                   //
// LogInterval(1,10000,5)                                                   //
//     i=0 --> 1              note: StepSize(ibin=0) =  not defined !!      //
//     i=1 --> 10                   StepSize(ibin=1) = 9                    //
//     i=2 --> 100                  StepSize(ibin=2) = 99                   //
//     i=3 --> 1000                 StepSize(ibin=3) = 999                  //
//     i=4 --> 10000                StepSize(ibin=4) = 9999                 //
//                                                                          //
// LogInterval(1,1000,11)                                                   //
//    i=0 --> 1                                                             //
//    i=1 --> 1.99526                                                       //
//    i=2 --> 3.98107                                                       //
//    i=3 --> 7.94328                                                       //
//    i=4 --> 15.8489                                                       //
//    i=5 --> 31.6228                                                       //
//    i=6 --> 63.0957                                                       //
//    i=7 --> 125.893                                                       //
//    i=8 --> 251.189                                                       //
//    i=9 --> 501.187                                                       //
//    i=10 --> 1000                                                         //
//                                                                          //
// LogInterval(1,1024,11)                                                   //
//    i=0 --> 1                                                             //
//    i=1 --> 2                                                             //
//    i=2 --> 4                                                             //
//    i=3 --> 8                                                             //
//    i=4 --> 16                                                            //
//    i=5 --> 32                                                            //
//    i=6 --> 64                                                            //
//    i=7 --> 128                                                           //
//    i=8 --> 256                                                           //
//    i=9 --> 512                                                           //
//    i=10 --> 1024                                                         //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
#include "Rtypes.h"

#include "Interval.h"



class TRandom3;

namespace TMVA {

   class MsgLogger;

   class LogInterval : public Interval {

   public:

      LogInterval( Double_t min, Double_t max, Int_t nbins = 0 );
      LogInterval( const LogInterval& other );
      virtual ~LogInterval();

      // accessors
      virtual Double_t GetMin()   const { return fMin; }
      virtual Double_t GetMax()   const { return fMax; }
      virtual Double_t GetWidth() const;
      virtual Int_t    GetNbins() const { return fNbins; }
      virtual Double_t GetMean()  const;
      virtual Double_t GetRndm( TRandom3& )  const;
      virtual Double_t GetElement( Int_t position ) const;
      virtual Double_t GetStepSize(Int_t iBin=0) const;

      void SetMax( Double_t m ) { fMax = m; }
      void SetMin( Double_t m ) { fMin = m; }

      MsgLogger& Log() const;

      ClassDef(Interval,0);    // Interval definition, continuous and discrete
   };

} // namespace TMVA

#endif
