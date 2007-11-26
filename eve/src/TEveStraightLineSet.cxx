// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveStraightLineSet.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TVirtualPad.h"
#include "TVirtualViewer3D.h"

#include "TRandom.h"
#include "TEveProjectionManager.h"

//______________________________________________________________________________
// TEveStraightLineSet
//
// Set of straight lines with optional markers along the lines.

ClassImp(TEveStraightLineSet)

//______________________________________________________________________________
TEveStraightLineSet::TEveStraightLineSet(const Text_t* n, const Text_t* t):
   TEveElement (),
   TNamed        (n, t),

   fLinePlex      (sizeof(TEveLine), 4),
   fMarkerPlex    (sizeof(Marker), 8),
   fOwnLinesIds   (kFALSE),
   fOwnMarkersIds (kFALSE),
   fRnrMarkers    (kTRUE),
   fRnrLines      (kTRUE),
   fLastLine      (0),
   fTrans         (kFALSE),
   fHMTrans       ()
{
   fMainColorPtr = &fLineColor;
   fLineColor    = 4;
   fMarkerColor  = 2;
   fMarkerStyle  = 20;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::AddLine(Float_t x1, Float_t y1, Float_t z1,
                                  Float_t x2, Float_t y2, Float_t z2)
{
   fLastLine = new (fLinePlex.NewAtom()) TEveLine(x1, y1, z1, x2, y2, z2);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::AddMarker(Int_t line, Float_t pos)
{
   /*Marker* marker = */new (fMarkerPlex.NewAtom()) Marker(line, pos);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::ComputeBBox()
{
   static const TEveException eH("TEveStraightLineSet::ComputeBBox ");
   if(fLinePlex.Size() == 0) {
      BBoxZero();
      return;
   }

   BBoxInit();

   TEveChunkManager::iterator li(fLinePlex);
   while (li.next()) {
      BBoxCheckPoint(((TEveLine*)li())->fV1);
      BBoxCheckPoint(((TEveLine*)li())->fV2);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::Paint(Option_t* /*option*/)
{
   static const TEveException eH("TEveStraightLineSet::Paint ");

   TBuffer3D buff(TBuffer3DTypes::kGeneric);

   // Section kCore
   buff.fID           = this;
   buff.fColor        = fLineColor;
   buff.fTransparency = 0;
   buff.fLocalFrame   = kFALSE;
   fHMTrans.SetBuffer3D(buff);
   buff.SetSectionsValid(TBuffer3D::kCore);

   Int_t reqSections = gPad->GetViewer3D()->AddObject(buff);
   if (reqSections != TBuffer3D::kNone)
      Error(eH, "only direct GL rendering supported.");
}

/******************************************************************************/

//______________________________________________________________________________
TClass* TEveStraightLineSet::ProjectedClass() const
{
   return TEveStraightLineSetProjected::Class();
}


//______________________________________________________________________________
// TEveStraightLineSetProjected
//
// Projected copy of a TEveStraightLineSet.

ClassImp(TEveStraightLineSetProjected)

//______________________________________________________________________________
   TEveStraightLineSetProjected::TEveStraightLineSetProjected() : TEveStraightLineSet(), TEveProjected ()
{}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetProjected::SetProjection(TEveProjectionManager* proj, TEveProjectable* model)
{
   TEveProjected::SetProjection(proj, model);

   // copy line and marker attributes
   * (TAttMarker*)this = * dynamic_cast<TAttMarker*>(fProjectable);
   * (TAttLine*)this   = * dynamic_cast<TAttLine*>(fProjectable);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetProjected::UpdateProjection()
{
   TEveProjection&   proj  = * fProjector->GetProjection();
   TEveStraightLineSet& orig  = * dynamic_cast<TEveStraightLineSet*>(fProjectable);

   // lines
   Int_t NL = orig.GetLinePlex().Size();
   fLinePlex.Reset(sizeof(TEveLine), NL);
   TEveLine* l;
   Float_t p1[3];
   Float_t p2[3];
   TEveChunkManager::iterator li(orig.GetLinePlex());

   Double_t s1, s2, s3;
   orig.RefHMTrans().GetScale(s1, s2, s3);
   TEveTrans mx; mx.Scale(s1, s2, s3);
   Double_t x, y, z;
   orig.RefHMTrans().GetPos(x, y,z);
   while (li.next())
   {
      l = (TEveLine*) li();
      p1[0] = l->fV1[0];  p1[1] = l->fV1[1]; p1[2] = l->fV1[2];
      p2[0] = l->fV2[0];  p2[1] = l->fV2[1]; p2[2] = l->fV2[2];
      mx.MultiplyIP(p1);
      mx.MultiplyIP(p2);
      p1[0] += x; p1[1] += y; p1[2] += z;
      p2[0] += x; p2[1] += y; p2[2] += z;
      proj.ProjectPointFv(p1);
      proj.ProjectPointFv(p2);
      AddLine(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
   }

   // markers
   Int_t NM = orig.GetMarkerPlex().Size();
   fMarkerPlex.Reset(sizeof(Marker), NM);
   Marker* m;
   TEveChunkManager::iterator mi(orig.GetMarkerPlex());
   while (mi.next())
   {
      m = (Marker*) mi();
      AddMarker(m->fLineID, m->fPos);
   }
}
