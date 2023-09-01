// @(#)root/geom:$Id$
// Author: Andrei Gheata   18/10/17

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TGeoRegion
\ingroup Geometry_classes

Regions are groups of volumes having a common set of user tracking cuts.

Class wrapper for regions used by Monte Carlo packages
A region is composed by a list of logical volumes and defines a set
of cuts. Used mainly to transport region information stored in
GDML format to the clients requiring it from the transient geometry.

*/

#include "TGeoRegion.h"

#include "TGeoManager.h"

ClassImp(TGeoRegionCut);
ClassImp(TGeoRegion);

////////////////////////////////////////////////////////////////////////////////
/// Region destructor.

TGeoRegion::~TGeoRegion()
{
   fCuts.Delete();
}

////////////////////////////////////////////////////////////////////////////////
/// Region copy constructor.
TGeoRegion::TGeoRegion(const TGeoRegion &other) : TNamed(other), fVolumes(other.fVolumes)
{
   for (int i = 0; i < other.GetNcuts(); ++i)
      AddCut(*other.GetCut(i));
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.
TGeoRegion &TGeoRegion::operator=(const TGeoRegion &other)
{
   if (&other != this) {
      TNamed::operator=(other);
      fVolumes.operator=(other.fVolumes);
      for (int i = 0; i < other.GetNcuts(); ++i)
         AddCut(*(TGeoRegionCut *)other.fCuts.At(i));
   }
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Add an existing volume to the region.
bool TGeoRegion::AddVolume(const char *name)
{
   if (!gGeoManager)
      return kFALSE;
   TGeoVolume *vol = gGeoManager->GetVolume(name);
   if (!vol)
      return kFALSE;
   AddVolume(vol);
   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// Add cut to the region.
void TGeoRegion::AddCut(const char *name, Double_t cut)
{
   fCuts.Add(new TGeoRegionCut(name, cut));
}

////////////////////////////////////////////////////////////////////////////////
/// Add an identical cut to the region.
void TGeoRegion::AddCut(const TGeoRegionCut &regioncut)
{
   fCuts.Add(new TGeoRegionCut(regioncut));
}

////////////////////////////////////////////////////////////////////////////////
/// Print region info
void TGeoRegion::Print(Option_t *) const
{
   printf("== Region: %s\n", GetName());
   printf("   volumes: ");
   for (int i = 0; i < GetNvolumes(); ++i)
      printf("%s ", GetVolume(i)->GetName());
   printf("\n");
   for (int i = 0; i < GetNcuts(); ++i)
      printf("   %s   value %g\n", GetCut(i)->GetName(), GetCut(i)->GetCut());
}
