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

   fLinePlex      (sizeof(Line_t), 4),
   fMarkerPlex    (sizeof(Marker_t), 8),
   fOwnLinesIds   (kFALSE),
   fOwnMarkersIds (kFALSE),
   fRnrMarkers    (kTRUE),
   fRnrLines      (kTRUE),
   fLastLine      (0),
   fTrans         (kFALSE),
   fHMTrans       ()
{
   // Constructor.

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
   // Add a line.

   fLastLine = new (fLinePlex.NewAtom()) Line_t(x1, y1, z1, x2, y2, z2);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::AddMarker(Int_t line, Float_t pos)
{
   // Add a marker for line with given index on relative position pos.

   /*Marker_t* marker = */new (fMarkerPlex.NewAtom()) Marker_t(line, pos);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::ComputeBBox()
{
   // Compute bounding-box.
   // Virtual from TAttBBox.

   static const TEveException eH("TEveStraightLineSet::ComputeBBox ");
   if(fLinePlex.Size() == 0) {
      BBoxZero();
      return;
   }

   BBoxInit();

   TEveChunkManager::iterator li(fLinePlex);
   while (li.next()) {
      BBoxCheckPoint(((Line_t*)li())->fV1);
      BBoxCheckPoint(((Line_t*)li())->fV2);
   }
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::Paint(Option_t* /*option*/)
{
   // Paint the line-set.

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
   // Return class of projected object.
   // Virtual from TEveProjectable.

   return TEveStraightLineSetProjected::Class();
}


//______________________________________________________________________________
// TEveStraightLineSetProjected
//
// Projected copy of a TEveStraightLineSet.

ClassImp(TEveStraightLineSetProjected)

//______________________________________________________________________________
TEveStraightLineSetProjected::TEveStraightLineSetProjected() :
   TEveStraightLineSet(), TEveProjected ()
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetProjected::SetProjection(TEveProjectionManager* proj,
                                                 TEveProjectable* model)
{
   // Set projection manager and model object.

   TEveProjected::SetProjection(proj, model);

   // copy line and marker attributes
   * (TAttMarker*)this = * dynamic_cast<TAttMarker*>(fProjectable);
   * (TAttLine*)  this = * dynamic_cast<TAttLine*>(fProjectable);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetProjected::UpdateProjection()
{
   // Callback that actually performs the projection.
   // Called when projection parameters have been updated.

   TEveProjection&      proj  = * fProjector->GetProjection();
   TEveStraightLineSet& orig  = * dynamic_cast<TEveStraightLineSet*>(fProjectable);

   // Lines
   Int_t NL = orig.GetLinePlex().Size();
   fLinePlex.Reset(sizeof(Line_t), NL);
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
      Line_t* l = (Line_t*) li();
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

   // Markers
   Int_t NM = orig.GetMarkerPlex().Size();
   fMarkerPlex.Reset(sizeof(Marker_t), NM);
   TEveChunkManager::iterator mi(orig.GetMarkerPlex());
   while (mi.next())
   {
      Marker_t* m = (Marker_t*) mi();
      AddMarker(m->fLineID, m->fPos);
   }
}
