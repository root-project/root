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

#include "TRandom.h"
#include "TEveProjectionManager.h"


//==============================================================================
//==============================================================================
// TEveStraightLineSet
//==============================================================================

//______________________________________________________________________________
//
// Set of straight lines with optional markers along the lines.

ClassImp(TEveStraightLineSet);

//______________________________________________________________________________
TEveStraightLineSet::TEveStraightLineSet(const char* n, const char* t):
   TEveElement (),
   TNamed      (n, t),

   fLinePlex      (sizeof(Line_t), 4),
   fMarkerPlex    (sizeof(Marker_t), 8),
   fOwnLinesIds   (kFALSE),
   fOwnMarkersIds (kFALSE),
   fRnrMarkers    (kTRUE),
   fRnrLines      (kTRUE),
   fDepthTest     (kTRUE),
   fLastLine      (0)
{
   // Constructor.

   InitMainTrans();
   fPickable = kTRUE;

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

//______________________________________________________________________________
void TEveStraightLineSet::AddMarker(Int_t line, Float_t pos)
{
   // Add a marker for line with given index on relative position pos.

   /*Marker_t* marker = */new (fMarkerPlex.NewAtom()) Marker_t(line, pos);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSet::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.

   const TEveStraightLineSet* m = dynamic_cast<const TEveStraightLineSet*>(el);
   if (m)
   {
      TAttLine::operator=(*m);
      TAttMarker::operator=(*m);
      fRnrMarkers = m->fRnrMarkers;
      fRnrLines   = m->fRnrLines;
      fDepthTest  = m->fDepthTest;
   }

   TEveElement::CopyVizParams(el);
}

//______________________________________________________________________________
void TEveStraightLineSet::WriteVizParams(ostream& out, const TString& var)
{
   // Write visualization parameters.

   TEveElement::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttMarker::SaveMarkerAttributes(out, var);
   TAttLine  ::SaveLineAttributes  (out, var);
   out << t << "SetRnrMarkers(" << ToString(fRnrMarkers) << ");\n";
   out << t << "SetRnrLines("   << ToString(fRnrLines)   << ");\n";
   out << t << "SetDepthTest("  << ToString(fDepthTest)  << ");\n";
}

/******************************************************************************/

//______________________________________________________________________________
TClass* TEveStraightLineSet::ProjectedClass(const TEveProjection*) const
{
   // Return class of projected object.
   // Virtual from TEveProjectable.

   return TEveStraightLineSetProjected::Class();
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
void TEveStraightLineSet::Paint(Option_t*)
{
   // Paint the line-set.

   PaintStandard(this);
}


//==============================================================================
//==============================================================================
// TEveStraightLineSetProjected
//==============================================================================

//______________________________________________________________________________
//
// Projected replica of a TEveStraightLineSet.

ClassImp(TEveStraightLineSetProjected);

//______________________________________________________________________________
TEveStraightLineSetProjected::TEveStraightLineSetProjected() :
   TEveStraightLineSet(), TEveProjected ()
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveStraightLineSetProjected::SetProjection(TEveProjectionManager* mng,
                                                 TEveProjectable* model)
{
   // Set projection manager and model object.

   TEveProjected::SetProjection(mng, model);

   // copy line and marker attributes
   * (TAttMarker*)this = * dynamic_cast<TAttMarker*>(fProjectable);
   * (TAttLine*)  this = * dynamic_cast<TAttLine*>(fProjectable);
}

//______________________________________________________________________________
void TEveStraightLineSetProjected::SetDepthLocal(Float_t d)
{
   // Set depth (z-coordinate) of the projected points.

   SetDepthCommon(d, this, fBBox);

   TEveChunkManager::iterator li(fLinePlex);
   while (li.next())
   {
      TEveStraightLineSet::Line_t& l = * (TEveStraightLineSet::Line_t*) li();
      l.fV1[2] = fDepth;
      l.fV2[2] = fDepth;
   }
}

//______________________________________________________________________________
void TEveStraightLineSetProjected::UpdateProjection()
{
   // Callback that actually performs the projection.
   // Called when projection parameters have been updated.

   TEveProjection&      proj = * fManager->GetProjection();
   TEveStraightLineSet& orig = * dynamic_cast<TEveStraightLineSet*>(fProjectable);

   BBoxClear();

   // Lines
   fLinePlex.Reset(sizeof(Line_t), orig.GetLinePlex().Size());
   Float_t p1[3];
   Float_t p2[3];
   TEveChunkManager::iterator li(orig.GetLinePlex());

   TEveTrans& origTrans = orig.RefMainTrans();
   Double_t s1, s2, s3;
   Double_t x, y, z;
   origTrans.GetScale(s1, s2, s3);
   origTrans.GetPos(x, y, z);

   TEveTrans mx;
   mx.Scale(s1, s2, s3);
   while (li.next())
   {
      Line_t* l = (Line_t*) li();
      p1[0] = l->fV1[0]; p1[1] = l->fV1[1]; p1[2] = l->fV1[2];
      p2[0] = l->fV2[0]; p2[1] = l->fV2[1]; p2[2] = l->fV2[2];
      mx.MultiplyIP(p1);
      mx.MultiplyIP(p2);
      p1[0] += x; p1[1] += y; p1[2] += z;
      p2[0] += x; p2[1] += y; p2[2] += z;
      proj.ProjectPointfv(p1, fDepth);
      proj.ProjectPointfv(p2, fDepth);
      AddLine(p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
   }

   // Markers
   fMarkerPlex.Reset(sizeof(Marker_t), orig.GetMarkerPlex().Size());
   TEveChunkManager::iterator mi(orig.GetMarkerPlex());
   while (mi.next())
   {
      Marker_t *m = (Marker_t*) mi();
      Line_t  *lo = (Line_t*) orig.GetLinePlex().Atom(m->fLineID);
      Line_t  *lp = (Line_t*) fLinePlex.Atom(m->fLineID);

      TEveVector t1, d, xx;

      t1.Set(lo->fV1); xx.Set(lo->fV2); xx -= t1; xx *= m->fPos; xx += t1;
      proj.ProjectVector(xx, 0);
      t1.Set(lp->fV1); d.Set(lp->fV2); d -= t1; xx -= t1;

      AddMarker(m->fLineID, d.Dot(xx) / d.Mag2());
   }
}
