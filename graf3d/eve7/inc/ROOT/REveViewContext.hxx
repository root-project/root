// @(#)root/eve7:$Id$
// Author: Matevz Tadel, Alja Tadel 2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveViewContext
#define ROOT7_REveViewContext

namespace ROOT {
namespace Experimental {

class REveTableViewInfo;
class REveTrackPropagator;

class REveViewContext  {
private:
   float m_R{100};
   float m_Z{100};
   REveTrackPropagator *m_trackPropagator{nullptr};
   REveTableViewInfo *fTableInfo{nullptr};

public:
   REveViewContext() = default;
   virtual ~REveViewContext() {}

   void SetBarrel(float r, float z) { m_R = r; m_Z = z; }

   void SetTrackPropagator(REveTrackPropagator *p) { m_trackPropagator = p; }
   void SetTableViewInfo(REveTableViewInfo *ti) { fTableInfo = ti; }

   float GetMaxR() const { return m_R; }
   float GetMaxZ() const { return m_Z; }
   REveTrackPropagator *GetPropagator() const { return m_trackPropagator; }
   REveTableViewInfo *GetTableViewInfo() const { return fTableInfo; }
};
}
}

#endif
