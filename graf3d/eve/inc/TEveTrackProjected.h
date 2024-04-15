// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveTrackProjected
#define ROOT_TEveTrackProjected

#include "TEveTrack.h"
#include "TEveProjectionBases.h"
#include <vector>

class TEveTrackProjected : public TEveTrack,
                           public TEveProjected
{
   friend class TEveTrackProjectedGL;

private:
   TEveTrackProjected(const TEveTrackProjected&) = delete;
   TEveTrackProjected& operator=(const TEveTrackProjected&) = delete;

   Int_t GetBreakPointIdx(Int_t start);

   TEveVector*          fOrigPnts;     // original track points

protected:
   std::vector<Int_t>   fBreakPoints; // indices of track break-points

   void SetDepthLocal(Float_t d) override;

public:
   TEveTrackProjected();
   ~TEveTrackProjected() override {}

   void SetProjection(TEveProjectionManager* mng, TEveProjectable* model) override;

   void UpdateProjection() override;
   TEveElement* GetProjectedAsElement() override { return this; }
   void MakeTrack(Bool_t recurse=kTRUE) override;


   void         PrintLineSegments();

   void SecSelected(TEveTrack*) override; // marked as signal in TEveTrack

   ClassDefOverride(TEveTrackProjected, 0); // Projected copy of a TEveTrack.
};


/******************************************************************************/
// TEveTrackListProjected
/******************************************************************************/

class TEveTrackListProjected : public TEveTrackList,
                               public TEveProjected
{
private:
   TEveTrackListProjected(const TEveTrackListProjected&);            // Not implemented
   TEveTrackListProjected& operator=(const TEveTrackListProjected&); // Not implemented

protected:
   void SetDepthLocal(Float_t d) override;

public:
   TEveTrackListProjected();
   ~TEveTrackListProjected() override {}

   void SetProjection(TEveProjectionManager* proj, TEveProjectable* model) override;
   void UpdateProjection() override  {}
   TEveElement* GetProjectedAsElement() override { return this; }

   void SetDepth(Float_t d) override;
   virtual void SetDepth(Float_t d, TEveElement* el);

   ClassDefOverride(TEveTrackListProjected, 0); // Specialization of TEveTrackList for holding TEveTrackProjected objects.
};

#endif
