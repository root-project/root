// @(#)root/tmva $Id$    
// Author: Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SVEvent                                                               *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Event class for Support Vector Machine                                    *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <azemla@cern.ch>        - IFJ PAN, Krakow, Poland          *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *   
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         * 
 *      MPI-K Heidelberg, Germany                                                 * 
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_SVEvent
#define ROOT_TMVA_SVEvent

#include <vector>
#include <iostream>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace TMVA 
{
   class Event;

   class SVEvent {

   public:

      SVEvent();
      SVEvent( const Event*, Float_t, Bool_t isSignal = kFALSE );
      SVEvent( const std::vector<Float_t>*, Float_t alpha, Int_t typeFlag, UInt_t ns );
      SVEvent( const std::vector<Float_t>* svector, Float_t alpha, Float_t alpha_p, Int_t typeFlag);

      virtual ~SVEvent();

      void SetAlpha        ( Float_t  alpha )      { fAlpha = alpha; }
      void SetAlpha_p      ( Float_t  alpha )      { fAlpha_p = alpha; }
      void SetErrorCache   ( Float_t  err_cache )  { fErrorCache = err_cache; }
      void SetIsShrinked   ( Int_t    isshrinked ) { fIsShrinked = isshrinked; }
      void SetLine         ( Float_t* line )       { fLine = line; } 
      void SetIdx          ( Int_t    idx )        { fIdx = idx; }
      void SetNs           ( UInt_t   ns )         { fNs = ns; }
      void UpdateErrorCache(Float_t upercache )    { fErrorCache += upercache; } 
         
      std::vector<Float_t>* GetDataVector() { return &fDataVector; } 
      Float_t  GetAlpha()         const { return fAlpha; }
      Float_t  GetAlpha_p()       const { return fAlpha_p; }
      Float_t  GetDeltaAlpha()    const { return fAlpha -  fAlpha_p; }

      Float_t  GetErrorCache()  const { return fErrorCache; }
      Int_t    GetTypeFlag()    const  { return fTypeFlag; }
      Int_t    GetNVar()        const  { return fNVar; }
      Int_t    GetIdx()         const  { return fIdx;}
      Float_t* GetLine()        const  { return fLine;}
      UInt_t   GetNs()          const  { return fNs;} 
      Float_t  GetCweight()     const  { return fCweight;}
      Float_t  GetTarget()      const  { return fTarget;}
      
      Bool_t  IsInI0a() const { return (0.< fAlpha) && (fAlpha<fCweight); }
      Bool_t  IsInI0b() const { return (0.< fAlpha) && (fAlpha_p<fCweight); }
      Bool_t  IsInI0()  const { return (IsInI0a() || IsInI0b()); } 
      Bool_t  IsInI1()  const { return (fAlpha == 0. && fAlpha_p == 0.); }
      Bool_t  IsInI2()  const { return (fAlpha == 0. && fAlpha_p == fCweight); }
      Bool_t  IsInI3()  const { return (fAlpha == fCweight && fAlpha_p == 0.); }
      
      void Print( std::ostream& os ) const;
      void PrintData();

   private:

      std::vector<Float_t> fDataVector;
      const Float_t        fCweight;     // svm cost weight
      Float_t              fAlpha;       // lagrange multiplier
      Float_t              fAlpha_p;     // lagrange multiplier
      Float_t              fErrorCache;  // optimization parameter
      UInt_t               fNVar;        // number of variables
      const Int_t          fTypeFlag;    // is sig or bkg - svm requieres 1 for sig and -1 for bkg
      Int_t                fIdx;         // index flag
      UInt_t               fNs;          // documentation
      Int_t                fIsShrinked;  // shrinking flag, see documentation
      Float_t*             fLine;        // pointer to column of kerenl matrix 
      const Float_t        fTarget;      // regression target
      
      ClassDef(SVEvent,0) // Event for SVM
   };
}

#endif //ROOT_TMVA_SVEvent
