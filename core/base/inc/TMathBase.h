// @(#)root/base:$Id$
// Authors: Rene Brun, Fons Rademakers   29/07/95

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMathBase
#define ROOT_TMathBase


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMath Base functions                                                 //
//                                                                      //
// Define the functions Min, Max, Abs, Sign, Range for all types.       //
// NB: These functions are unfortunately not available in a portable    //
// way in std::.                                                        //
//                                                                      //
// More functions are defined in TMath.h. TMathBase.h is designed to be //
// a stable file and used in place of TMath.h in the ROOT miniCore.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

namespace TMath {

   // Abs
   inline Short_t  Abs(Short_t d);
   inline Int_t    Abs(Int_t d);
   inline Long_t   Abs(Long_t d);
   inline Long64_t Abs(Long64_t d);
   inline Float_t  Abs(Float_t d);
   inline Double_t Abs(Double_t d);

   // Even/Odd
   inline Bool_t   Even(Long_t a);
   inline Bool_t   Odd(Long_t a);

   // Sign
   inline Short_t  Sign(Short_t a, Short_t b);
   inline Int_t    Sign(Int_t a, Int_t b);
   inline Long_t   Sign(Long_t a, Long_t b);
   inline Long64_t Sign(Long64_t a, Long64_t b);
   inline Float_t  Sign(Float_t a, Float_t b);
   inline Double_t Sign(Double_t a, Double_t b);

   // Min, Max of two scalars
   inline Short_t   Min(Short_t a, Short_t b);
   inline UShort_t  Min(UShort_t a, UShort_t b);
   inline Int_t     Min(Int_t a, Int_t b);
   inline UInt_t    Min(UInt_t a, UInt_t b);
   inline Long_t    Min(Long_t a, Long_t b);
   inline ULong_t   Min(ULong_t a, ULong_t b);
   inline Long64_t  Min(Long64_t a, Long64_t b);
   inline ULong64_t Min(ULong64_t a, ULong64_t b);
   inline Float_t   Min(Float_t a, Float_t b);
   inline Double_t  Min(Double_t a, Double_t b);

   inline Short_t   Max(Short_t a, Short_t b);
   inline UShort_t  Max(UShort_t a, UShort_t b);
   inline Int_t     Max(Int_t a, Int_t b);
   inline UInt_t    Max(UInt_t a, UInt_t b);
   inline Long_t    Max(Long_t a, Long_t b);
   inline ULong_t   Max(ULong_t a, ULong_t b);
   inline Long64_t  Max(Long64_t a, Long64_t b);
   inline ULong64_t Max(ULong64_t a, ULong64_t b);
   inline Float_t   Max(Float_t a, Float_t b);
   inline Double_t  Max(Double_t a, Double_t b);

   // Range
   inline Short_t   Range(Short_t lb, Short_t ub, Short_t x);
   inline Int_t     Range(Int_t lb, Int_t ub, Int_t x);
   inline Long_t    Range(Long_t lb, Long_t ub, Long_t x);
   inline ULong_t   Range(ULong_t lb, ULong_t ub, ULong_t x);
   inline Double_t  Range(Double_t lb, Double_t ub, Double_t x);

   //NextPrime is used by the Core classes.
   Long_t   NextPrime(Long_t x);   // Least prime number greater than x
}


//---- Even/odd ----------------------------------------------------------------

inline Bool_t TMath::Even(Long_t a)
   { return ! (a & 1); }

inline Bool_t TMath::Odd(Long_t a)
   { return (a & 1); }

//---- Abs ---------------------------------------------------------------------

inline Short_t TMath::Abs(Short_t d)
{ return (d >= 0) ? d : Short_t(-d); }

inline Int_t TMath::Abs(Int_t d)
   { return (d >= 0) ? d : -d; }

inline Long_t TMath::Abs(Long_t d)
   { return (d >= 0) ? d : -d; }

inline Long64_t TMath::Abs(Long64_t d)
   { return (d >= 0) ? d : -d; }

inline Float_t TMath::Abs(Float_t d)
   { return (d >= 0) ? d : -d; }

inline Double_t TMath::Abs(Double_t d)
   { return (d >= 0) ? d : -d; }

//---- Sign --------------------------------------------------------------------

inline Short_t TMath::Sign(Short_t a, Short_t b)
{ return (b >= 0) ? Abs(a) : Short_t(-Abs(a)); }

inline Int_t TMath::Sign(Int_t a, Int_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Long_t TMath::Sign(Long_t a, Long_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Long64_t TMath::Sign(Long64_t a, Long64_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Float_t TMath::Sign(Float_t a, Float_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

inline Double_t TMath::Sign(Double_t a, Double_t b)
   { return (b >= 0) ? Abs(a) : -Abs(a); }

//---- Min ---------------------------------------------------------------------

inline Short_t TMath::Min(Short_t a, Short_t b)
   { return a <= b ? a : b; }

inline UShort_t TMath::Min(UShort_t a, UShort_t b)
   { return a <= b ? a : b; }

inline Int_t TMath::Min(Int_t a, Int_t b)
   { return a <= b ? a : b; }

inline UInt_t TMath::Min(UInt_t a, UInt_t b)
   { return a <= b ? a : b; }

inline Long_t TMath::Min(Long_t a, Long_t b)
   { return a <= b ? a : b; }

inline ULong_t TMath::Min(ULong_t a, ULong_t b)
   { return a <= b ? a : b; }

inline Long64_t TMath::Min(Long64_t a, Long64_t b)
   { return a <= b ? a : b; }

inline ULong64_t TMath::Min(ULong64_t a, ULong64_t b)
   { return a <= b ? a : b; }

inline Float_t TMath::Min(Float_t a, Float_t b)
   { return a <= b ? a : b; }

inline Double_t TMath::Min(Double_t a, Double_t b)
   { return a <= b ? a : b; }

//---- Max ---------------------------------------------------------------------

inline Short_t TMath::Max(Short_t a, Short_t b)
   { return a >= b ? a : b; }

inline UShort_t TMath::Max(UShort_t a, UShort_t b)
   { return a >= b ? a : b; }

inline Int_t TMath::Max(Int_t a, Int_t b)
   { return a >= b ? a : b; }

inline UInt_t TMath::Max(UInt_t a, UInt_t b)
   { return a >= b ? a : b; }

inline Long_t TMath::Max(Long_t a, Long_t b)
   { return a >= b ? a : b; }

inline ULong_t TMath::Max(ULong_t a, ULong_t b)
   { return a >= b ? a : b; }

inline Long64_t TMath::Max(Long64_t a, Long64_t b)
   { return a >= b ? a : b; }

inline ULong64_t TMath::Max(ULong64_t a, ULong64_t b)
   { return a >= b ? a : b; }

inline Float_t TMath::Max(Float_t a, Float_t b)
   { return a >= b ? a : b; }

inline Double_t TMath::Max(Double_t a, Double_t b)
   { return a >= b ? a : b; }

//---- Range -------------------------------------------------------------------

inline Short_t TMath::Range(Short_t lb, Short_t ub, Short_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Int_t TMath::Range(Int_t lb, Int_t ub, Int_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Long_t TMath::Range(Long_t lb, Long_t ub, Long_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline ULong_t TMath::Range(ULong_t lb, ULong_t ub, ULong_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }

inline Double_t TMath::Range(Double_t lb, Double_t ub, Double_t x)
   { return x < lb ? lb : (x > ub ? ub : x); }


#endif
