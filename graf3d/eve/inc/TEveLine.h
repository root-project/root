// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveLine
#define ROOT_TEveLine

#include "TEvePointSet.h"
#include "TEveVector.h"

#include "TAttLine.h"

//------------------------------------------------------------------------------
// TEveLine
//------------------------------------------------------------------------------

class TEveLine : public TEvePointSet,
                 public TAttLine
{
   friend class TEveLineEditor;
   friend class TEveLineGL;

private:
   TEveLine(const TEveLine&);            // Not implemented
   TEveLine& operator=(const TEveLine&); // Not implemented

protected:
   Bool_t  fRnrLine;
   Bool_t  fRnrPoints;
   Bool_t  fSmooth;

   static Bool_t fgDefaultSmooth;

public:
   TEveLine(Int_t n_points=0, ETreeVarType_e tv_type=kTVT_XYZ);
   TEveLine(const char* name, Int_t n_points=0, ETreeVarType_e tv_type=kTVT_XYZ);
   ~TEveLine() override {}

   void SetMarkerColor(Color_t col) override;

   void SetLineColor(Color_t col) override   { SetMainColor(col); }
   void SetLineStyle(Style_t lstyle) override;
   void SetLineWidth(Width_t lwidth) override;

   Bool_t GetRnrLine() const     { return fRnrLine;   }
   Bool_t GetRnrPoints() const   { return fRnrPoints; }
   Bool_t GetSmooth() const      { return fSmooth;    }
   void   SetRnrLine(Bool_t r);
   void   SetRnrPoints(Bool_t r);
   void   SetSmooth(Bool_t r);

   void    ReduceSegmentLengths(Float_t max);
   Float_t CalculateLineLength() const;

   TEveVector GetLineStart() const;
   TEveVector GetLineEnd()   const;

   const TGPicture* GetListTreeIcon(Bool_t open=kFALSE) override;

   void CopyVizParams(const TEveElement* el) override;
   void WriteVizParams(std::ostream& out, const TString& var) override;

   TClass* ProjectedClass(const TEveProjection* p) const override;

   static Bool_t GetDefaultSmooth();
   static void   SetDefaultSmooth(Bool_t r);

   ClassDefOverride(TEveLine, 0); // An arbitrary polyline with fixed line and marker attributes.
};


//------------------------------------------------------------------------------
// TEveLineProjected
//------------------------------------------------------------------------------

class TEveLineProjected : public TEveLine,
                          public TEveProjected
{
private:
   TEveLineProjected(const TEveLineProjected&);            // Not implemented
   TEveLineProjected& operator=(const TEveLineProjected&); // Not implemented

protected:
   void SetDepthLocal(Float_t d) override;

public:
   TEveLineProjected();
   ~TEveLineProjected() override {}

   void SetProjection(TEveProjectionManager* mng, TEveProjectable* model) override;
   void UpdateProjection() override;
   TEveElement* GetProjectedAsElement() override { return this; }

   ClassDefOverride(TEveLineProjected, 0); // Projected replica of a TEveLine.
};

#endif
