// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveLine
#define ROOT7_REveLine

#include <ROOT/REvePointSet.hxx>
#include <ROOT/REveVector.hxx>

#include "TAttLine.h"

namespace ROOT {
namespace Experimental {

//------------------------------------------------------------------------------
// REveLine
//------------------------------------------------------------------------------

class REveLine : public REvePointSet,
                 public TAttLine
{
private:
   REveLine &operator=(const REveLine &); // Not implemented

protected:
   Bool_t fRnrLine;
   Bool_t fRnrPoints;
   Bool_t fSmooth;

   static Bool_t fgDefaultSmooth;

public:
   REveLine(const char *name="", const char *title="", Int_t n_points = 0);
   REveLine(const REveLine &l);
   virtual ~REveLine() {}

   virtual void SetMarkerColor(Color_t col);

   virtual void SetLineColor(Color_t col) { SetMainColor(col); }
   virtual void SetLineStyle(Style_t lstyle);
   virtual void SetLineWidth(Width_t lwidth);

   Bool_t GetRnrLine() const { return fRnrLine; }
   Bool_t GetRnrPoints() const { return fRnrPoints; }
   Bool_t GetSmooth() const { return fSmooth; }
   void   SetRnrLine(Bool_t r);
   void   SetRnrPoints(Bool_t r);
   void   SetSmooth(Bool_t r);

   void    ReduceSegmentLengths(Float_t max);
   Float_t CalculateLineLength() const;

   REveVector GetLineStart() const;
   REveVector GetLineEnd() const;

   virtual void CopyVizParams(const REveElement *el);
   virtual void WriteVizParams(std::ostream &out, const TString &var);

   virtual TClass *ProjectedClass(const REveProjection *p) const;

   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset); // override
   void BuildRenderData();                                    // override {}

   static Bool_t GetDefaultSmooth();
   static void SetDefaultSmooth(Bool_t r);

   ClassDef(REveLine, 0); // An arbitrary polyline with fixed line and marker attributes.
};

//------------------------------------------------------------------------------
// REveLineProjected
//------------------------------------------------------------------------------

class REveLineProjected : public REveLine, public REveProjected {
private:
   REveLineProjected(const REveLineProjected &);            // Not implemented
   REveLineProjected &operator=(const REveLineProjected &); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   REveLineProjected();
   virtual ~REveLineProjected() {}

   virtual void SetProjection(REveProjectionManager *mng, REveProjectable *model);
   virtual void UpdateProjection();
   virtual REveElement *GetProjectedAsElement() { return this; }

   ClassDef(REveLineProjected, 0); // Projected replica of a REveLine.
};

} // namespace Experimental
} // namespace ROOT

#endif
