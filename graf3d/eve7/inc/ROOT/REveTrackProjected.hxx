// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveTrackProjected
#define ROOT7_REveTrackProjected

#include <ROOT/REveTrack.hxx>
#include <ROOT/REveProjectionBases.hxx>

namespace ROOT {
namespace Experimental {

class REveTrackProjected : public REveTrack,
                           public REveProjected
{
private:
   REveTrackProjected(const REveTrackProjected &);            // Not implemented
   REveTrackProjected &operator=(const REveTrackProjected &); // Not implemented

   Int_t GetBreakPointIdx(Int_t start);

   REveVector *fOrigPnts{nullptr}; // original track points

protected:
   std::vector<Int_t> fBreakPoints; // indices of track break-points

   virtual void SetDepthLocal(Float_t d);

public:
   REveTrackProjected();
   virtual ~REveTrackProjected() {}

   virtual void SetProjection(REveProjectionManager *mng, REveProjectable *model);

   virtual void UpdateProjection();
   virtual REveElement *GetProjectedAsElement() { return this; }
   virtual void MakeTrack(Bool_t recurse = kTRUE);

   void PrintLineSegments();

   virtual void SecSelected(REveTrack *); // marked as signal in REveTrack

   Int_t WriteCoreJson(nlohmann::json &cj, Int_t rnr_offset); // override
   void BuildRenderData();                                    // override;

   ClassDef(REveTrackProjected, 0); // Projected copy of a REveTrack.
};

/******************************************************************************/
// REveTrackListProjected
/******************************************************************************/

class REveTrackListProjected : public REveTrackList, public REveProjected {
private:
   REveTrackListProjected(const REveTrackListProjected &);            // Not implemented
   REveTrackListProjected &operator=(const REveTrackListProjected &); // Not implemented

protected:
   virtual void SetDepthLocal(Float_t d);

public:
   REveTrackListProjected();
   virtual ~REveTrackListProjected() {}

   virtual void SetProjection(REveProjectionManager *proj, REveProjectable *model);
   virtual void UpdateProjection() {}
   virtual REveElement *GetProjectedAsElement() { return this; }

   virtual void SetDepth(Float_t d);
   virtual void SetDepth(Float_t d, REveElement *el);

   ClassDef(REveTrackListProjected, 0); // Specialization of REveTrackList for holding REveTrackProjected objects.
};

} // namespace Experimental
} // namespace ROOT

#endif
