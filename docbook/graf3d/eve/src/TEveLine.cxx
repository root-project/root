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
   fMarkerColor  =  kGreen;
}

//______________________________________________________________________________
TEveLine::TEveLine(const char* name, Int_t n_points, ETreeVarType_e tv_type) :
   TEvePointSet(name, n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   // Constructor.

   fMainColorPtr = &fLineColor;
   fMarkerColor = kGreen;
}

//______________________________________________________________________________
const TGPicture* TEveLine::GetListTreeIcon(Bool_t)
{
   // Returns list-tree icon for TEveLine.

   return fgListTreeIcons[8];
}

//______________________________________________________________________________
void TEveLine::SetMarkerColor(Color_t col)
{
   // Set marker color. Propagate to projected lines.

   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* l = dynamic_cast<TEveLine*>(*pi);
      if (l && fMarkerColor == l->GetMarkerColor())
      {
         l->SetMarkerColor(col);
         l->StampObjProps();
      }
      ++pi;
   }
   TAttMarker::SetMarkerColor(col);
}

//______________________________________________________________________________
void TEveLine::SetLineStyle(Style_t lstyle)
{
   // Set line-style of the line.
   // The style is propagated to projecteds.

   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* pt = dynamic_cast<TEveLine*>(*pi);
      if (pt)
      {
         pt->SetLineStyle(lstyle);
         pt->StampObjProps();
      }
      ++pi;
   }
   TAttLine::SetLineStyle(lstyle);
}

//______________________________________________________________________________
void TEveLine::SetLineWidth(Width_t lwidth)
{
   // Set line-style of the line.
   // The style is propagated to projecteds.

   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* pt = dynamic_cast<TEveLine*>(*pi);
      if (pt)
      {
         pt->SetLineWidth(lwidth);
         pt->StampObjProps();
      }
      ++pi;
   }
   TAttLine::SetLineWidth(lwidth);
}

//______________________________________________________________________________
void TEveLine::SetRnrLine(Bool_t r)
{
   // Set rendering of line. Propagate to projected lines.

   fRnrLine = r;
   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* l = dynamic_cast<TEveLine*>(*pi);
      if (l)
      {
         l->SetRnrLine(r);
         l->ElementChanged();
      }
      ++pi;
   }
}

//______________________________________________________________________________
void TEveLine::SetRnrPoints(Bool_t r)
{
   // Set rendering of points. Propagate to projected lines.

   fRnrPoints = r;
   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* l = dynamic_cast<TEveLine*>(*pi);
      if (l)
      {
         l->SetRnrPoints(r);
         l->ElementChanged();
      }
      ++pi;
   }
}

//______________________________________________________________________________
void TEveLine::SetSmooth(Bool_t r)
{
   // Set smooth rendering. Propagate to projected lines.

   fSmooth = r;
   std::list<TEveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      TEveLine* l = dynamic_cast<TEveLine*>(*pi);
      if (l)
      {
         l->SetSmooth(r);
         l->ElementChanged();
      }
      ++pi;
   }
}

//==============================================================================

//______________________________________________________________________________
void TEveLine::ReduceSegmentLengths(Float_t max)
{
   // Make sure that no segment is longer than max.
   // Per point references and integer ids are lost.

   const Float_t max2 = max*max;

   Float_t    *p = GetP();
   Int_t       s = Size();
   TEveVector  a, b, d;

   std::vector<TEveVector> q;

   b.Set(p);
   q.push_back(b);
   for (Int_t i = 1; i < s; ++i)
   {
      a = b; b.Set(&p[3*i]); d = b - a;
      Float_t m2 = d.Mag2();
      if (m2 > max2)
      {
         Float_t f = TMath::Sqrt(m2) / max;
         Int_t   n = TMath::FloorNint(f);
         d *= 1.0f / (n + 1);
         for (Int_t j = 0; j < n; ++j)
         {
            a += d;
            q.push_back(a);
         }
      }
      q.push_back(b);
   }

   s = q.size();
   Reset(s);
   for (std::vector<TEveVector>::iterator i = q.begin(); i != q.end(); ++i)
      SetNextPoint(i->fX, i->fY, i->fZ);
}

//______________________________________________________________________________
TEveVector TEveLine::GetLineStart() const
{
   // Return the first point of the line.
   // If there are no points (0,0,0) is returned.

   TEveVector v;
   GetPoint(0, v.fX, v.fY, v.fZ);
   return v;
}

//______________________________________________________________________________
TEveVector TEveLine::GetLineEnd() const
{
   // Return the last point of the line.
   // If there are no points (0,0,0) is returned.

   TEveVector v;
   GetPoint(fLastPoint, v.fX, v.fY, v.fZ);
   return v;
}

//==============================================================================

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
   out << t << "SetRnrLine("   << ToString(fRnrLine)   << ");\n";
   out << t << "SetRnrPoints(" << ToString(fRnrPoints) << ");\n";
   out << t << "SetSmooth("    << ToString(fSmooth)    << ");\n";
}

//______________________________________________________________________________
TClass* TEveLine::ProjectedClass(const TEveProjection*) const
{
   // Virtual from TEveProjectable, returns TEvePointSetProjected class.

   return TEveLineProjected::Class();
}

//------------------------------------------------------------------------------

//______________________________________________________________________________
Bool_t TEveLine::GetDefaultSmooth()
{
   // Get default value for smooth-line drawing flag.
   // Static function.

   return fgDefaultSmooth;
}

//______________________________________________________________________________
void TEveLine::SetDefaultSmooth(Bool_t r)
{
   // Set default value for smooth-line drawing flag (default kFALSE).
   // Static function.

   fgDefaultSmooth = r;
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
   CopyVizParams(dynamic_cast<TEveElement*>(model));
}

//______________________________________________________________________________
void TEveLineProjected::SetDepthLocal(Float_t d)
{
   // Set depth (z-coordinate) of the projected points.

   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP() + 2;
   for (Int_t i = 0; i < n; ++i, p+=3)
      *p = fDepth;
}

//______________________________________________________________________________
void TEveLineProjected::UpdateProjection()
{
   // Re-apply the projection.
   // Virtual from TEveProjected.

   TEveProjection& proj = * fManager->GetProjection();
   TEveLine      & als  = * dynamic_cast<TEveLine*>(fProjectable);
   TEveTrans      *tr   =   als.PtrMainTrans(kFALSE);

   Int_t n = als.Size();
   Reset(n);
   fLastPoint = n - 1;
   Float_t *o = als.GetP(), *p = GetP();
   for (Int_t i = 0; i < n; ++i, o+=3, p+=3)
   {
      proj.ProjectPointfv(tr, o, p, fDepth);
   }
}
