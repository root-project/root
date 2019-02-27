// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_REveStraightLineSet
#define ROOT_REveStraightLineSet

#include "TNamed.h"
#include "TAttMarker.h"
#include "TAttLine.h"
#include "TAttBBox.h"

#include "ROOT/REveElement.hxx"
#include "ROOT/REveProjectionBases.hxx"
#include "ROOT/REveChunkManager.hxx"
#include "ROOT/REveTrans.hxx"

class TRandom;

namespace ROOT {
namespace Experimental {

//------------------------------------------------------------------------------
// REveStraightLineSet
//------------------------------------------------------------------------------
class REveStraightLineSet : public REveElement,
                            public REveProjectable,
                            public TAttLine,
                            public TAttMarker,
                            public TAttBBox
{
private:
   REveStraightLineSet(const REveStraightLineSet&);            // Not implemented
   REveStraightLineSet& operator=(const REveStraightLineSet&); // Not implemented

public:
   struct Line_t
   {
      Int_t          fId;
      Float_t        fV1[3];
      Float_t        fV2[3];

      Line_t(Float_t x1, Float_t y1, Float_t z1,
             Float_t x2, Float_t y2, Float_t z2) : fId(-1)
      {
         fV1[0] = x1, fV1[1] = y1, fV1[2] = z1;
         fV2[0] = x2, fV2[1] = y2, fV2[2] = z2;
      }
   };

   struct Marker_t
   {
      Float_t      fV[3];
      Int_t        fLineId;

      Marker_t(Float_t x, Float_t y, Float_t z, Int_t line_id) : fLineId(line_id)
      {
         fV[0] = x, fV[1] = y, fV[2] = z;
      }
   };

protected:
   REveChunkManager  fLinePlex;
   REveChunkManager  fMarkerPlex;

   Bool_t            fOwnLinesIds;    // Flag specifying if id-objects are owned by the line-set
   Bool_t            fOwnMarkersIds;  // Flag specifying if id-objects are owned by the line-set

   Bool_t            fRnrMarkers;
   Bool_t            fRnrLines;

   Bool_t            fDepthTest;

   Line_t*           fLastLine; //!

public:
   REveStraightLineSet(const std::string& n="StraightLineSet", const std::string& t="");
   virtual ~REveStraightLineSet() {}

   virtual void SetLineColor(Color_t col) { SetMainColor(col); }

   Line_t*   AddLine(Float_t x1, Float_t y1, Float_t z1, Float_t x2, Float_t y2, Float_t z2);
   Line_t*   AddLine(const REveVector& p1, const REveVector& p2);
   Marker_t* AddMarker(Float_t x, Float_t y, Float_t z, Int_t line_id=-1);
   Marker_t* AddMarker(const REveVector& p, Int_t line_id=-1);
   Marker_t* AddMarker(Int_t line_id, Float_t pos);

   void      SetLine(int idx, Float_t x1, Float_t y1, Float_t z1, Float_t x2, Float_t y2, Float_t z2);
   void      SetLine(int idx, const REveVector& p1, const REveVector& p2);

   REveChunkManager& GetLinePlex()   { return fLinePlex;   }
   REveChunkManager& GetMarkerPlex() { return fMarkerPlex; }

   virtual Bool_t GetRnrMarkers() { return fRnrMarkers; }
   virtual Bool_t GetRnrLines()   { return fRnrLines;   }
   virtual Bool_t GetDepthTest()  { return fDepthTest;   }

   virtual void SetRnrMarkers(Bool_t x) { fRnrMarkers = x; }
   virtual void SetRnrLines(Bool_t x)   { fRnrLines   = x; }
   virtual void SetDepthTest(Bool_t x)  { fDepthTest   = x; }

   virtual void CopyVizParams(const REveElement* el);
   virtual void WriteVizParams(std::ostream& out, const TString& var);

   virtual TClass* ProjectedClass(const REveProjection* p) const; // override;

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset); // override;
   void  BuildRenderData(); // override;

   void ComputeBBox(); // override;

   ClassDef(REveStraightLineSet, 0); // Set of straight lines with optional markers along the lines.
};


//==============================================================================
// REveStraightLineSetProjected
//==============================================================================

class REveStraightLineSetProjected : public REveStraightLineSet,
                                     public REveProjected
{
private:
   REveStraightLineSetProjected(const REveStraightLineSetProjected&);            // Not implemented
   REveStraightLineSetProjected& operator=(const REveStraightLineSetProjected&); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   REveStraightLineSetProjected();
   virtual ~REveStraightLineSetProjected() {}

   virtual void SetProjection(REveProjectionManager* mng, REveProjectable* model);
   virtual void UpdateProjection();
   virtual REveElement* GetProjectedAsElement() { return this; }

   ClassDef(REveStraightLineSetProjected, 0); // Projected copy of a REveStraightLineSet.
};

} // namespace Experimental
} // namespace ROOT

#endif
