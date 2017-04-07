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

namespace
{
   inline Float_t sqr(Float_t x) { return x*x; }
}

/** \class TEveLine
\ingroup TEve
An arbitrary polyline with fixed line and marker attributes.
*/

ClassImp(TEveLine);

Bool_t TEveLine::fgDefaultSmooth = kFALSE;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveLine::TEveLine(Int_t n_points, ETreeVarType_e tv_type) :
   TEvePointSet("Line", n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   fMainColorPtr = &fLineColor;
   fMarkerColor  =  kGreen;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveLine::TEveLine(const char* name, Int_t n_points, ETreeVarType_e tv_type) :
   TEvePointSet(name, n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   fMainColorPtr = &fLineColor;
   fMarkerColor = kGreen;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns list-tree icon for TEveLine.

const TGPicture* TEveLine::GetListTreeIcon(Bool_t)
{
   return fgListTreeIcons[8];
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker color. Propagate to projected lines.

void TEveLine::SetMarkerColor(Color_t col)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set line-style of the line.
/// The style is propagated to projecteds.

void TEveLine::SetLineStyle(Style_t lstyle)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set line-style of the line.
/// The style is propagated to projecteds.

void TEveLine::SetLineWidth(Width_t lwidth)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of line. Propagate to projected lines.

void TEveLine::SetRnrLine(Bool_t r)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set rendering of points. Propagate to projected lines.

void TEveLine::SetRnrPoints(Bool_t r)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Set smooth rendering. Propagate to projected lines.

void TEveLine::SetSmooth(Bool_t r)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Make sure that no segment is longer than max.
/// Per point references and integer ids are lost.

void TEveLine::ReduceSegmentLengths(Float_t max)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Sum-up lengths of individual segments.

Float_t TEveLine::CalculateLineLength() const
{
   Float_t sum = 0;

   Int_t    s = Size();
   Float_t *p = GetP();
   for (Int_t i = 1; i < s; ++i, p += 3)
   {
      sum += TMath::Sqrt(sqr(p[3] - p[0]) + sqr(p[4] - p[1]) + sqr(p[5] - p[2]));
   }
   return sum;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the first point of the line.
/// If there are no points (0,0,0) is returned.

TEveVector TEveLine::GetLineStart() const
{
   TEveVector v;
   GetPoint(0, v.fX, v.fY, v.fZ);
   return v;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the last point of the line.
/// If there are no points (0,0,0) is returned.

TEveVector TEveLine::GetLineEnd() const
{
   TEveVector v;
   GetPoint(fLastPoint, v.fX, v.fY, v.fZ);
   return v;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.

void TEveLine::CopyVizParams(const TEveElement* el)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Write visualization parameters.

void TEveLine::WriteVizParams(std::ostream& out, const TString& var)
{
   TEvePointSet::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttLine::SaveLineAttributes(out, var);
   out << t << "SetRnrLine("   << ToString(fRnrLine)   << ");\n";
   out << t << "SetRnrPoints(" << ToString(fRnrPoints) << ");\n";
   out << t << "SetSmooth("    << ToString(fSmooth)    << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveProjectable, returns TEvePointSetProjected class.

TClass* TEveLine::ProjectedClass(const TEveProjection*) const
{
   return TEveLineProjected::Class();
}

////////////////////////////////////////////////////////////////////////////////
/// Get default value for smooth-line drawing flag.
/// Static function.

Bool_t TEveLine::GetDefaultSmooth()
{
   return fgDefaultSmooth;
}

////////////////////////////////////////////////////////////////////////////////
/// Set default value for smooth-line drawing flag (default kFALSE).
/// Static function.

void TEveLine::SetDefaultSmooth(Bool_t r)
{
   fgDefaultSmooth = r;
}

/** \class TEveLineProjected
\ingroup TEve
Projected copy of a TEveLine.
*/

ClassImp(TEveLineProjected);

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TEveLineProjected::TEveLineProjected() :
   TEveLine      (),
   TEveProjected ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection manager and projection model.
/// Virtual from TEveProjected.

void TEveLineProjected::SetProjection(TEveProjectionManager* mng,
                                      TEveProjectable* model)
{
   TEveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<TEveElement*>(model));
}

////////////////////////////////////////////////////////////////////////////////
/// Set depth (z-coordinate) of the projected points.

void TEveLineProjected::SetDepthLocal(Float_t d)
{
   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP() + 2;
   for (Int_t i = 0; i < n; ++i, p+=3)
      *p = fDepth;
}

////////////////////////////////////////////////////////////////////////////////
/// Re-apply the projection.
/// Virtual from TEveProjected.

void TEveLineProjected::UpdateProjection()
{
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
