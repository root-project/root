// @(#)root/geom:$Id$
// Author: Andrei Gheata   18/10/17
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGeoRegion
#define ROOT_TGeoRegion

#include "TNamed.h"

#include "TObjArray.h"

#include "TGeoVolume.h"

class TGeoRegionCut : public TNamed {
protected:
   Double_t fCut{0.}; // Cut value

public:
   TGeoRegionCut() {}
   TGeoRegionCut(const char *name, Double_t cut) : TNamed(name, ""), fCut(cut) {}

   ~TGeoRegionCut() override {}

   Double_t GetCut() const { return fCut; }
   void SetCut(Double_t cut) { fCut = cut; }

   ClassDefOverride(TGeoRegionCut, 1) // A region cut
};

class TGeoRegion : public TNamed {
protected:
   TObjArray fVolumes; // list of volumes in this region
   TObjArray fCuts;    // list of cuts for the region

public:
   TGeoRegion() {}
   TGeoRegion(const char *name, const char *title = "") : TNamed(name, title) {}
   TGeoRegion(const TGeoRegion &other);
   TGeoRegion &operator=(const TGeoRegion &other);
   ~TGeoRegion() override;

   // Volume accessors
   void AddVolume(TGeoVolume *vol) { fVolumes.Add(vol); }
   bool AddVolume(const char *name);
   int GetNvolumes() const { return fVolumes.GetEntriesFast(); }
   TGeoVolume *GetVolume(int i) const { return (TGeoVolume *)fVolumes.At(i); }

   // Cuts accessors
   void AddCut(const char *name, Double_t cut);
   void AddCut(const TGeoRegionCut &regioncut);
   int GetNcuts() const { return fCuts.GetEntriesFast(); }
   TGeoRegionCut *GetCut(int i) const { return (TGeoRegionCut *)fCuts.At(i); }

   void Print(Option_t *option = "") const override; // *MENU*

   ClassDefOverride(TGeoRegion, 1) // Region wrapper class
};

#endif
