/* @(#)root/mathcore:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ nestedtypedefs;
#pragma link C++ nestedclasses;

#pragma link C++ namespace TMath;

#pragma link C++ global ROOT_TMath;

#pragma link C++ class TComplex+;

#pragma link C++ function operator*(Double_t, const TComplex&);
#pragma link C++ function operator+(Double_t, const TComplex&);
#pragma link C++ function operator/(Double_t, const TComplex&);
#pragma link C++ function operator-(Double_t, const TComplex&);
#pragma link C++ function operator>>(istream&,TComplex&);
#pragma link C++ function operator<<(ostream&,const TComplex&);

#pragma link C++ function TMath::Limits<Double_t>::Min();
#pragma link C++ function TMath::Limits<Double_t>::Max();
#pragma link C++ function TMath::Limits<Double_t>::Epsilon();

#pragma link C++ function TMath::Limits<Float_t>::Min();
#pragma link C++ function TMath::Limits<Float_t>::Max();
#pragma link C++ function TMath::Limits<Float_t>::Epsilon();

#pragma link C++ function TMath::Limits<Int_t>::Max();
#pragma link C++ function TMath::Limits<Int_t>::Min();

// exclude these since they do not work in CINT
//#pragma link C++ function TMath::Limits<Long64_t>::Max();
//#pragma link C++ function TMath::Limits<Long64_t>::Min();

#pragma link C++ function TMath::MinElement(Long64_t, const Short_t*);
#pragma link C++ function TMath::MinElement(Long64_t, const Int_t*);
#pragma link C++ function TMath::MinElement(Long64_t, const Float_t*);
#pragma link C++ function TMath::MinElement(Long64_t, const Double_t*);
#pragma link C++ function TMath::MinElement(Long64_t, const Long_t*);
#pragma link C++ function TMath::MinElement(Long64_t, const Long64_t*);

#pragma link C++ function TMath::MaxElement(Long64_t, const Short_t*);
#pragma link C++ function TMath::MaxElement(Long64_t, const Int_t*);
#pragma link C++ function TMath::MaxElement(Long64_t, const Float_t*);
#pragma link C++ function TMath::MaxElement(Long64_t, const Double_t*);
#pragma link C++ function TMath::MaxElement(Long64_t, const Long_t*);
#pragma link C++ function TMath::MaxElement(Long64_t, const Long64_t*);

#pragma link C++ function TMath::LocMin(Long64_t, const Short_t*);
#pragma link C++ function TMath::LocMin(Long64_t, const Int_t*);
#pragma link C++ function TMath::LocMin(Long64_t, const Float_t*);
#pragma link C++ function TMath::LocMin(Long64_t, const Double_t*);
#pragma link C++ function TMath::LocMin(Long64_t, const Long_t*);
#pragma link C++ function TMath::LocMin(Long64_t, const Long64_t*);

#pragma link C++ function TMath::LocMax(Long64_t, const Short_t*);
#pragma link C++ function TMath::LocMax(Long64_t, const Int_t*);
#pragma link C++ function TMath::LocMax(Long64_t, const Float_t*);
#pragma link C++ function TMath::LocMax(Long64_t, const Double_t*);
#pragma link C++ function TMath::LocMax(Long64_t, const Long_t*);
#pragma link C++ function TMath::LocMax(Long64_t, const Long64_t*);

#pragma link C++ function TMath::Mean(Long64_t, const Short_t*, const Double_t*);
#pragma link C++ function TMath::Mean(Long64_t, const Int_t*, const Double_t*);
#pragma link C++ function TMath::Mean(Long64_t, const Float_t*, const Double_t*);
#pragma link C++ function TMath::Mean(Long64_t, const Double_t*, const Double_t*);
#pragma link C++ function TMath::Mean(Long64_t, const Long_t*, const Double_t*);
#pragma link C++ function TMath::Mean(Long64_t, const Long64_t*, const Double_t*);

#pragma link C++ function TMath::GeomMean(Long64_t, const Short_t*);
#pragma link C++ function TMath::GeomMean(Long64_t, const Int_t*);
#pragma link C++ function TMath::GeomMean(Long64_t, const Float_t*);
#pragma link C++ function TMath::GeomMean(Long64_t, const Double_t*);
#pragma link C++ function TMath::GeomMean(Long64_t, const Long_t*);
#pragma link C++ function TMath::GeomMean(Long64_t, const Long64_t*);

#pragma link C++ function TMath::RMS(Long64_t, const Short_t*);
#pragma link C++ function TMath::RMS(Long64_t, const Int_t*);
#pragma link C++ function TMath::RMS(Long64_t, const Float_t*);
#pragma link C++ function TMath::RMS(Long64_t, const Double_t*);
#pragma link C++ function TMath::RMS(Long64_t, const Long_t*);
#pragma link C++ function TMath::RMS(Long64_t, const Long64_t*);

#pragma link C++ function TMath::Median(Long64_t, const Short_t*,  const Double_t*, Long64_t*);
#pragma link C++ function TMath::Median(Long64_t, const Int_t*,  const Double_t*, Long64_t*);
#pragma link C++ function TMath::Median(Long64_t, const Float_t*,  const Double_t*, Long64_t*);
#pragma link C++ function TMath::Median(Long64_t, const Double_t*,  const Double_t*, Long64_t*);
#pragma link C++ function TMath::Median(Long64_t, const Long_t*,  const Double_t*, Long64_t*);
#pragma link C++ function TMath::Median(Long64_t, const Long64_t*,  const Double_t*, Long64_t*);

#pragma link C++ function TMath::KOrdStat(Long64_t, const Short_t*, Long64_t, Long64_t*);
#pragma link C++ function TMath::KOrdStat(Long64_t, const Int_t*, Long64_t, Long64_t*);
#pragma link C++ function TMath::KOrdStat(Long64_t, const Float_t*, Long64_t, Long64_t*);
#pragma link C++ function TMath::KOrdStat(Long64_t, const Double_t*, Long64_t, Long64_t*);
#pragma link C++ function TMath::KOrdStat(Long64_t, const Long_t*, Long64_t, Long64_t*);
#pragma link C++ function TMath::KOrdStat(Long64_t, const Long64_t*, Long64_t, Long64_t*);

#pragma link C++ function TMath::BinarySearch(Long64_t, const Short_t*, Short_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Int_t*, Int_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Float_t*, Float_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Double_t*, Double_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Long_t*, Long_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Long64_t*, Long64_t);

#pragma link C++ function TMath::BinarySearch(Long64_t, const Short_t**, Short_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Int_t**, Int_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Float_t**, Float_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Double_t**, Double_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Long_t**, Long_t);
#pragma link C++ function TMath::BinarySearch(Long64_t, const Long64_t**, Long64_t);

#pragma link C++ function TMath::IsInside(Float_t, Float_t, Int_t, Float_t*, Float_t*);
#pragma link C++ function TMath::IsInside(Int_t, Int_t, Int_t, Int_t*, Int_t*);

#pragma link C++ function TMath::Sort(Long64_t, const Short_t*, Long64_t*, Bool_t);
#pragma link C++ function TMath::Sort(Long64_t, const Int_t*, Long64_t*, Bool_t);
#pragma link C++ function TMath::Sort(Long64_t, const Float_t*, Long64_t*, Bool_t);
#pragma link C++ function TMath::Sort(Long64_t, const Double_t*, Long64_t*, Bool_t);
#pragma link C++ function TMath::Sort(Long64_t, const Long_t*, Long64_t*, Bool_t);
#pragma link C++ function TMath::Sort(Long64_t, const Long64_t*, Long64_t*, Bool_t);

#pragma link C++ function TMath::Sort(Int_t, const Short_t*, Int_t*, Bool_t);
#pragma link C++ function TMath::Sort(Int_t, const Int_t*, Int_t*, Bool_t);
#pragma link C++ function TMath::Sort(Int_t, const Float_t*, Int_t*, Bool_t);
#pragma link C++ function TMath::Sort(Int_t, const Double_t*, Int_t*, Bool_t);
#pragma link C++ function TMath::Sort(Int_t, const Long_t*, Int_t*, Bool_t);
#pragma link C++ function TMath::Sort(Int_t, const Long64_t*, Int_t*, Bool_t);


#pragma link C++ function TMath::Cross(const Float_t*,const Float_t*, Float_t*);
#pragma link C++ function TMath::Cross(const Double_t*,const Double_t*, Double_t*);

#pragma link C++ function TMath::NormCross(const Float_t*,const Float_t*,Float_t*);
#pragma link C++ function TMath::NormCross(const Double_t*,const Double_t*,Double_t*);

#pragma link C++ function TMath::Normal2Plane(const Float_t*,const Float_t*,const Float_t*, Float_t*);
#pragma link C++ function TMath::Normal2Plane(const Double_t*,const Double_t*,const Double_t*, Double_t*);

#endif
