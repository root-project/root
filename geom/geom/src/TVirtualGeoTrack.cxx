// @(#)root/geom:$Id$
// Author: Andrei Gheata  2003/04/10

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGeoManager.h"

#include "TVirtualGeoTrack.h"

//______________________________________________________________________________
// TVirtualGeoTrack - Base class for user-defined tracks attached to a geometry.
//             Tracks are 3D objects made of points and they store a 
//             pointer to a TParticle. The geometry manager holds a list
//             of all tracks that will be deleted on destruction of 
//             gGeoManager.
//
//______________________________________________________________________________

ClassImp(TVirtualGeoTrack)

//______________________________________________________________________________
TVirtualGeoTrack::TVirtualGeoTrack()
{
//*-*-*-*-*-*-*-*-*-*-*Virtual tracks default constructor*-*-*-*-*-*-*-*-*
//*-*                  ==================================
   fPDG        = 0;
   fId         = -1;
   fParent     = 0;
   fParticle   = 0;
   fTracks     = 0;
}

//______________________________________________________________________________
TVirtualGeoTrack::TVirtualGeoTrack(Int_t id, Int_t pdgcode, TVirtualGeoTrack *parent, TObject *particle)
{
// Constructor providing ID for parent track (-1 for primaries), ID of this
// track and related particle pointer.
   fPDG        = pdgcode;
   fId         = id;
   fParent     = parent;
   fParticle   = particle;
   fTracks     = 0;
}

//_____________________________________________________________________________
TVirtualGeoTrack::TVirtualGeoTrack(const TVirtualGeoTrack& other)
                 :TObject(other), TGeoAtt(other), TAttLine(other), TAttMarker(other),
                  fPDG(other.fPDG),
                  fId(other.fId),
                  fParent(other.fParent),
                  fParticle(other.fParticle),
                  fTracks(other.fTracks)
{
// Copy ctor. NOT TO BE CALLED.
}

//_____________________________________________________________________________
TVirtualGeoTrack& TVirtualGeoTrack::operator=(const TVirtualGeoTrack& gv) 
{
   // Assignment operator. NOT TO BE CALLED.
   if(this!=&gv) {
      TObject::operator=(gv);
      TGeoAtt::operator=(gv);
      TAttLine::operator=(gv);
      TAttMarker::operator=(gv);
      fPDG=gv.fPDG;
      fId=gv.fId;
      fParent=gv.fParent;
      fParticle=gv.fParticle;
      fTracks=gv.fTracks;
   } 
   return *this;
}

//______________________________________________________________________________
TVirtualGeoTrack::~TVirtualGeoTrack()
{
// Destructor.
   if (fTracks) {
      fTracks->Delete();
      delete fTracks;
   }   
}

//______________________________________________________________________________
TVirtualGeoTrack *TVirtualGeoTrack::FindTrackWithId(Int_t id) const
{
// Recursively search through this track for a daughter
// particle (at any depth) with the specified id
   TVirtualGeoTrack* trk=0;
   if (GetId()==id) {
      trk = (TVirtualGeoTrack*)this;
      return trk;
   }
   TVirtualGeoTrack* kid=0;
   Int_t nd = GetNdaughters();
   for (Int_t i=0; i<nd; i++) if (GetDaughterId(i) == id) return GetDaughter(i);
   for (Int_t i=0; i<nd; i++) {
      kid = GetDaughter(i);
      if (kid!=0) {
         trk = kid->FindTrackWithId(id);
         if (trk!=0) break;
      }
   }
   return trk;
}

//______________________________________________________________________________
const char *TVirtualGeoTrack::GetName() const
{
// Get the PDG name.
   return gGeoManager->GetPdgName(fPDG);
}

//______________________________________________________________________________
Bool_t TVirtualGeoTrack::IsInTimeRange() const
{
// True if track TOF range overlaps with time interval of TGeoManager
   Double_t tmin, tmax;
   Bool_t timecut = gGeoManager->GetTminTmax(tmin,tmax);
   if (!timecut) return kTRUE;
   const Double_t *point = GetFirstPoint();
   if (!point) return kFALSE;
   if (point[3]>tmax) return kFALSE;
   point = GetLastPoint();
   if (point[3]<tmin) return kFALSE;
   return kTRUE;
}     

//______________________________________________________________________________
void TVirtualGeoTrack::SetName(const char *name)
{
// Set a default name for this track.
   gGeoManager->SetPdgName(fPDG, name);
   if (!strcmp(name, "gamma")) {
      SetLineColor(kGreen);
      SetMarkerColor(kGreen);
      SetLineWidth(1);
      SetLineStyle(kDotted);
      return;
   }     
   if (!strcmp(name, "pi+") || !strcmp(name, "proton") || !strcmp(name, "K+")) {
      SetLineColor(kRed);
      SetMarkerColor(kRed);
      SetLineWidth(2);
      return;
   }     
   if (!strcmp(name, "pi-") || !strcmp(name, "K-")) {
      SetLineColor(30);
      SetMarkerColor(30);
      SetLineWidth(2);
      return;
   }     
   if (!strcmp(name, "pi0") || !strcmp(name, "K0")) {
      SetLineColor(kCyan);
      SetMarkerColor(kCyan);
      SetLineWidth(2);
      return;
   }     
   if (!strcmp(name, "neutron")) {
      SetLineColor(16);
      SetMarkerColor(16);
      SetLineWidth(1);
      SetLineStyle(kDotted);
      return;
   }     
   if (!strcmp(name, "Alpha") || !strcmp(name, "Deuteron") || !strcmp(name, "Triton")) {
      SetLineColor(kMagenta);
      SetMarkerColor(kMagenta);
      SetLineWidth(3);
      return;
   }     
   if (!strcmp(name, "e-") || !strcmp(name, "mu-")) {
      SetLineColor(kBlue);
      SetMarkerColor(kBlue);
      SetLineWidth(1);
      SetLineStyle(kDotted);
      return;
   }     
   if (!strcmp(name, "e+") || !strcmp(name, "mu+")) {
      SetLineColor(kMagenta);
      SetMarkerColor(kMagenta);
      SetLineWidth(1);
      SetLineStyle(kDotted);
      return;
   }     
}

   
