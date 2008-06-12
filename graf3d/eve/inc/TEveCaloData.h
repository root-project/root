// @(#)root/eve:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveCaloData
#define ROOT_TEveCaloData

#include <vector>
#include "Rtypes.h"
#include "TEveUtil.h"

class TH2F;
class TAxis;
class THStack;

class TEveCaloData: public TEveRefBackPtr
{
public:
   struct SliceInfo_t
   {
      TString  fName;
      Float_t  fThreshold;
      Int_t    fID;
      Color_t  fColor;
      TH2F    *fHist;

      SliceInfo_t(): fName(""), fThreshold(0), fID(-1), fColor(Color_t(4)), fHist(0){}
      SliceInfo_t(TH2F* h): fName(""), fThreshold(0), fID(-1), fColor(Color_t(4)), fHist(h) {}

      virtual ~SliceInfo_t() {}

      void Setup(const char* name, Float_t threshold, Color_t col)
      {
         fName      = name;
         fThreshold = threshold;
         fColor     = col;
      };

      ClassDef(SliceInfo_t, 0); // Slice info for histogram stack.
   };

   typedef std::vector<SliceInfo_t>           vSliceInfo_t;
   typedef std::vector<SliceInfo_t>::iterator vSliceInfo_i;

   /**************************************************************************/

   struct CellId_t
   {
      // Cell ID inner structure.

      Int_t fTower;
      Int_t fSlice;

      CellId_t(Int_t t, Int_t s):fTower(t), fSlice(s){}
   };

   struct CellData_t
   {
      // Cell geometry inner structure.

      Float_t fValue;
      Float_t fPhiMin;
      Float_t fPhiMax;
      Float_t fThetaMin;  // theta=0 on z axis
      Float_t fThetaMax;
      Float_t fEtaMin;
      Float_t fEtaMax;
      Int_t   fZSideSign;

      CellData_t(): fValue(0), fPhiMin(0), fPhiMax(0), fThetaMin(0), fThetaMax(0), fZSideSign(1) {}
      void Configure(Float_t v, Float_t e1, Float_t e2, Float_t p1, Float_t p2);

      Float_t Value(Bool_t)    const;

      Float_t EtaMin()   const { return fEtaMin; }
      Float_t EtaMax()   const { return fEtaMax; }
      Float_t Eta()      const { return (fEtaMin+fEtaMax)*0.5f; }
      Float_t EtaDelta() const { return fEtaMax-fEtaMin; }

      Float_t ThetaMin() const { return fThetaMin; }
      Float_t ThetaMax() const { return fThetaMax; }
      Float_t Theta() const { return (fThetaMax+fThetaMin)*0.5f; }
      Float_t ThetaDelta() const { return fThetaMax-fThetaMin; }

      Float_t PhiMin()   const { return fPhiMin; }
      Float_t PhiMax()   const { return fPhiMax; }
      Float_t Phi()      const { return (fPhiMin+fPhiMax)*0.5f; }
      Float_t PhiDelta() const { return fPhiMax-fPhiMin; }

      Float_t ZSideSign()const { return fZSideSign;}

      void Dump() const;
   };

   typedef std::vector<CellId_t>           vCellId_t;
   typedef std::vector<CellId_t>::iterator vCellId_i;

private:
   TEveCaloData(const TEveCaloData&);            // Not implemented
   TEveCaloData& operator=(const TEveCaloData&); // Not implemented

protected:
   vSliceInfo_t fSliceInfos;

   TAxis*       fEtaAxis;
   TAxis*       fPhiAxis;

public:
   TEveCaloData();
   virtual ~TEveCaloData(){}

   virtual void GetCellList(Float_t etaMin, Float_t etaMax,
                            Float_t phi, Float_t phiRng, vCellId_t &out) const = 0;

   virtual void GetCellData(const CellId_t &id, CellData_t& data) const = 0;
   virtual void GetCellData(const CellId_t &id, Float_t  phiMin, Float_t phiRng, CellData_t& data) const = 0;

   virtual void  InvalidateUsersCache() = 0;

   virtual Bool_t SupportsEtaBinning(){ return kFALSE; }
   virtual Bool_t SupportsPhiBinning(){ return kFALSE; }

   virtual TAxis* GetEtaBins(){ return fEtaAxis;} 
   virtual TAxis* GetPhiBins(){ return fPhiAxis ;}

   virtual Int_t    GetNSlices() const = 0; 
   virtual Float_t  GetMaxVal(Bool_t et) const = 0;
   SliceInfo_t&     RefSliceInfo(Int_t s) { return fSliceInfos[s]; }

   virtual void  GetEtaLimits(Double_t &min, Double_t &max) const = 0;
   virtual void  GetPhiLimits(Double_t &min, Double_t &max) const = 0;


   ClassDef(TEveCaloData, 0); // Manages calorimeter event data.
};

/**************************************************************************/
/**************************************************************************/

class TEveCaloDataHist: public TEveCaloData
{
private:
   TEveCaloDataHist(const TEveCaloDataHist&);            // Not implemented
   TEveCaloDataHist& operator=(const TEveCaloDataHist&); // Not implemented

protected:
   THStack*    fHStack;

   Float_t     fMaxValEt; // cached
   Float_t     fMaxValE;  // cached

public:
   TEveCaloDataHist();
   virtual ~TEveCaloDataHist();

   virtual void GetCellList( Float_t etaMin, Float_t etaMax,
                             Float_t phi, Float_t phiRng, vCellId_t &out) const;

   virtual void GetCellData(const TEveCaloData::CellId_t &id, TEveCaloData::CellData_t& data) const;
   virtual void GetCellData(const CellId_t &id, Float_t  phiMin, Float_t phiRng, CellData_t& data) const;

   virtual void  InvalidateUsersCache();

   virtual Bool_t SupportsEtaBinning(){ return kTRUE; }
   virtual Bool_t SupportsPhiBinning(){ return kTRUE; }

   virtual void    GetEtaLimits(Double_t &min, Double_t &max) const;
   virtual void    GetPhiLimits(Double_t &min, Double_t &max) const;

   Int_t    AddHistogram(TH2F* hist);
   virtual Float_t GetMaxVal(Bool_t et) const {return (et)? fMaxValEt:fMaxValE;}

   virtual Int_t   GetNSlices() const;

   ClassDef(TEveCaloDataHist, 0); // Manages calorimeter TH2F event data.
};

#endif
