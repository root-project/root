// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveLine.h"
#include "TEveProjectionManager.h"

//==============================================================================
//==============================================================================
// TEveLine
//==============================================================================

//______________________________________________________________________________
//
// An arbitrary polyline with fixed line and marker attributes.

ClassImp(TEveLine);

Bool_t TEveLine::fgDefaultSmooth = kFALSE;

//______________________________________________________________________________
TEveLine::TEveLine(Int_t n_points, ETreeVarType_e tv_type) :
   TEvePointSet("Line", n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   // Constructor.

   fMainColorPtr = &fLineColor;
}

//______________________________________________________________________________
TEveLine::TEveLine(const Text_t* name, Int_t n_points, ETreeVarType_e tv_type) :
   TEvePointSet(name, n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   // Constructor.

   fMainColorPtr = &fLineColor;
}

//______________________________________________________________________________
void TEveLine::CopyVizParams(const TEveElement* el)
{
   // Copy visualization parameters from element el.

   const TEveLine* m = dynamic_cast<const TEveLine*>(el);
   if (m)
   {
      TAttLine::operator=(*m);
      fRnrLine   = m->fRnrLine;
      fRnrPoints = m->fRnrPoints;
      fSmooth    = m->fSmooth;
   }

   TEvePointSet::CopyVizParams(el);
}

//______________________________________________________________________________
void TEveLine::WriteVizParams(ostream& out, const TString& var)
{
   // Write visualization parameters.

   TEvePointSet::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttLine::SaveLineAttributes(out, var);
   out << t << "SetRnrLine("   << fRnrLine   << ");\n";
   out << t << "SetRnrPoints(" << fRnrPoints << ");\n";
   out << t << "SetSmooth("    << fSmooth    << ");\n";
}

//______________________________________________________________________________
TClass* TEveLine::ProjectedClass() const
{
   // Virtual from TEveProjectable, returns TEvePointSetProjected class.

   return TEveLineProjected::Class();
}

//==============================================================================
//==============================================================================
// TEveLineProjected
//==============================================================================

//______________________________________________________________________________
//
// Projected copy of a TEvePointSet.

ClassImp(TEveLineProjected);

//______________________________________________________________________________
TEveLineProjected::TEveLineProjected() :
   TEveLine      (),
   TEveProjected ()
{
   // Default constructor.
}

//______________________________________________________________________________
void TEveLineProjected::SetProjection(TEveProjectionManager* mng,
                                      TEveProjectable* model)
{
   // Set projection manager and projection model.
   // Virtual from TEveProjected.

   TEveProjected::SetProjection(mng, model);
   * (TAttMarker*)this = * dynamic_cast<TAttMarker*>(fProjectable);
   * (TAttLine*)  this = * dynamic_cast<TAttLine*>  (fProjectable);
}

//______________________________________________________________________________
void TEveLineProjected::SetDepth(Float_t d)
{
   // Set depth (z-coordinate) of the projected points.

   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP();
   for (Int_t i = 0; i < n; ++i, p+=3)
      p[2] = fDepth;
}

//______________________________________________________________________________
void TEveLineProjected::UpdateProjection()
{
   // Re-apply the projection.
   // Virtual from TEveProjected.

   TEveProjection& proj = * fManager->GetProjection();
   TEveLine      & als   = * dynamic_cast<TEveLine*>(fProjectable);

   Int_t n = als.Size();
   Reset(n);
   fLastPoint = n - 1;
   Float_t *o = als.GetP(), *p = GetP();
   for (Int_t i = 0; i < n; ++i, o+=3, p+=3)
   {
      p[0] = o[0]; p[1] = o[1]; p[2] = o[2];
      proj.ProjectPoint(p[0], p[1], p[2]);
      p[2] = fDepth;
   }
}
