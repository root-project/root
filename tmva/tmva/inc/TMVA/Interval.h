// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : Interval                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *    Generic range definition (used, eg, in genetic algorithm)                   *
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

#ifndef ROOT_TMVA_Interval
#define ROOT_TMVA_Interval

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// Interval                                                                 //
//                                                                          //
// Interval definition, continuous and discrete                             //
//                                                                          //
// Interval(min,max)  : a continuous interval [min,max]                     //
// Interval(min,max,n): a "discrete interval" [min,max], i.e the n numbers: //
//          min, min+step, min+2*step,...., min+(n-1)*step, min+n*step=max  //
//   e.g.: Interval(1,5,5)=1,2,3,4,5                                        //
//         Interval(.5,1.,6)= .5, .6., .7, .8, .9, 1.0                      //
//                                                                          //
//  Note: **bin** counting starts from ZERO unlike in ROOT histograms       //
//                                                                          //
//    Example:   Interval(.5,1.,6)                                          //
//                                                                          //
//             [ min                           max ]                        //
//         ------------------------------------------------------------     //
//                |     |     |     |     |     |                           //
//               .5    .6    .7    .8    .9    1.0                          //
//                                                                          //
//         bin    0     1     2     3     4     5                           //
//                                                                          //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////
#include "Rtypes.h"

class TRandom3;

namespace TMVA {

   class MsgLogger;

   class Interval {

   public:

      Interval( Double_t min, Double_t max, Int_t nbins = 0 );
      Interval( const Interval& other );
      virtual ~Interval();

      // accessors
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

      virtual void Print( std::ostream& os ) const;

   protected:

      Double_t fMin, fMax;    ///< the constraints of the Interval
      Int_t    fNbins;        ///< when >0 : number of bins (discrete interval); when ==0 continuous interval

   private:
      MsgLogger& Log() const;

      ClassDef(Interval,0);    // Interval definition, continuous and discrete
   };

} // namespace TMVA

#endif
