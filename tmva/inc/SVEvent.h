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
      SVEvent( const Event*, const Float_t );
      SVEvent( const std::vector<Float_t>*, Float_t alpha, const Int_t typeFlag, const UInt_t ns );
      SVEvent( const std::vector<Float_t>* svector, Float_t alpha, Float_t alpha_p);

      virtual ~SVEvent();

      void SetAlpha        ( Float_t  alpha )      { fAlpha = alpha; }
      void SetAlpha_p	   ( Float_t  alpha )      { fAlpha_p = alpha; }
      void SetErrorCache   ( Float_t  err_cache )  { fErrorCache = err_cache; }
      void SetIsShrinked   ( Int_t    isshrinked ) { fIsShrinked = isshrinked; }
      void SetLine         ( Float_t* line )       { fLine = line; } 
      void SetIdx          ( Int_t    idx )        { fIdx = idx; }
      void SetNs           ( UInt_t   ns )         { fNs = ns; }
      void UpdateErrorCache(Float_t upercache )	   { fErrorCache += upercache; } 
         
      std::vector<Float_t>* GetDataVector()	{ return &fDataVector; } 
      Float_t  GetAlpha()		  const	{ return fAlpha; }
      Float_t  GetAlpha_p()		  const	{ return fAlpha_p; }
      Float_t  GetDeltaAlpha()		  const	{ return fAlpha -  fAlpha_p; }

      Float_t  GetErrorCache()  const 	{ return fErrorCache; }
      Int_t    GetTypeFlag()    const  { return fTypeFlag; }
      Int_t    GetNVar()        const  { return fNVar; }
      Int_t    GetIdx()         const  { return fIdx;}
      Float_t* GetLine()        const  { return fLine;}
      UInt_t   GetNs()          const  { return fNs;} 
      Float_t  GetCweight()     const  { return fCweight;}
      Float_t  GetTarget()	const  { return fTarget;}
      
            
      // TODO finde nicer solution!!!
      Bool_t  IsInI0a() const { return (0.< fAlpha) && (fAlpha<fCweight); }
      Bool_t  IsInI0b() const { return (0.< fAlpha) && (fAlpha_p<fCweight); }
      Bool_t  IsInI0() 	const { return (IsInI0a() || IsInI0b()); } 
      Bool_t  IsInI1() 	const { return (fAlpha == 0. && fAlpha_p == 0.); }
      Bool_t  IsInI2()	const { return (fAlpha == 0. && fAlpha_p == fCweight); }
      Bool_t  IsInI3()	const { return (fAlpha == fCweight && fAlpha_p == 0.); }
      //I0b = {i: 0< a_i_p < C_i}
      //I1  = {i: a_i && a_i_p ==0}
      //I2  = {i: a_i == 0,   a_i_p == C_i }
      //I3  = {i: a_i == C_i, a_i_p == 0   }
      
      void Print( std::ostream& os ) const;
      void PrintData();

   private:

      std::vector<Float_t> fDataVector;
      const Float_t fCweight;     // documentation
      Float_t 	     fAlpha;       // documentation
      Float_t	     fAlpha_p;		 // documentation
      Float_t 	     fErrorCache;  // documentation
			
      UInt_t 	     fNVar;        // documentation
      const Int_t   fTypeFlag;    // documentation
      Int_t         fIdx;         // documentation
      UInt_t        fNs;          // documentation
      Int_t 		  fIsShrinked;  // documentation
      Float_t*      fLine;        // documentation
      
      //for regression
      const Float_t 	fTarget;	//regression target
      //I0b = {i: 0< a_i_p < C_i}
      //I1  = {i: a_i && a_i_p ==0}
      //I2  = {i: a_i == 0,   a_i_p == C_i }
      //I3  = {i: a_i == C_i, a_i_p == 0   }
      
      ClassDef(SVEvent,0) // Event for SVM
   };
}

#endif //ROOT_TMVA_SVEvent
