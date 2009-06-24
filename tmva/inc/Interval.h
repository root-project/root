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

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Interval                                                             //
//                                                                      //
// Interval definition, continuous and discrete                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TRandom3;

namespace TMVA {

   class MsgLogger;
   
   class Interval {

   public:

      Interval( Double_t min, Double_t max, Int_t nbins = 0 );
      Interval( const Interval& other );
      virtual ~Interval();

      // accessors
      Double_t GetMin()   const { return fMin; }
      Double_t GetMax()   const { return fMax; }
      Double_t GetWidth() const { return fMax - fMin; }
      Int_t    GetNbins() const { return fNbins; }
      Double_t GetMean()  const { return (fMax + fMin)/2; }
      Double_t GetRndm( TRandom3& )  const;
      Double_t GetElement( Int_t position ) const;      

      void SetMax( Double_t m ) { fMax = m; }
      void SetMin( Double_t m ) { fMin = m; }

   private:

      Double_t fMin, fMax;    // the constraints of the Interval
      Int_t    fNbins;        // when >0 : number of bins (discrete interval); when =0 continuous interval

      mutable MsgLogger* fLogger;   // message logger
      MsgLogger& Log() const { return *fLogger; }          

      ClassDef(Interval,0)    // Interval definition, continous and discrete
   };

} // namespace TMVA

#endif
