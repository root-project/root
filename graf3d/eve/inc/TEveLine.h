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

#include "TEveUtil.h"
#include "TEvePointSet.h"

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
   virtual ~TEveLine() {}

   virtual void SetLineColor(Color_t col)   { SetMainColor(col); }
   virtual void SetMarkerColor(Color_t col);

   Bool_t GetRnrLine() const     { return fRnrLine;   }
   Bool_t GetRnrPoints() const   { return fRnrPoints; }
   Bool_t GetSmooth() const      { return fSmooth;    }
   void   SetRnrLine(Bool_t r);
   void   SetRnrPoints(Bool_t r);
   void   SetSmooth(Bool_t r);

   void   ReduceSegmentLengths(Float_t max);

   virtual void CopyVizParams(const TEveElement* el);
   virtual void WriteVizParams(ostream& out, const TString& var);

   virtual TClass* ProjectedClass() const;

   static Bool_t GetDefaultSmooth()       { return fgDefaultSmooth; }
   static void SetDefaultSmooth(Bool_t r) { fgDefaultSmooth = r;    }

   ClassDef(TEveLine, 0); // An arbitrary polyline with fixed line and marker attributes.
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

public:
   TEveLineProjected();
   virtual ~TEveLineProjected() {}

   virtual void SetProjection(TEveProjectionManager* mng, TEveProjectable* model);
   virtual void SetDepth(Float_t d);
   virtual void UpdateProjection();

   ClassDef(TEveLineProjected, 0); // Projected replica of a TEveLine.
};

#endif
