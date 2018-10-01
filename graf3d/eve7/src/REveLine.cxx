// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/REveLine.hxx"
#include "ROOT/REveProjectionManager.hxx"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

namespace
{
   inline Float_t sqr(Float_t x) { return x*x; }
}

/** \class REveLine
\ingroup REve
An arbitrary polyline with fixed line and marker attributes.
*/

Bool_t REveLine::fgDefaultSmooth = kFALSE;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveLine::REveLine(Int_t n_points, ETreeVarType_e tv_type) :
   REvePointSet("Line", n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   fMainColorPtr = &fLineColor;
   fMarkerColor  =  kGreen;
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveLine::REveLine(const char* name, Int_t n_points, ETreeVarType_e tv_type) :
   REvePointSet(name, n_points, tv_type),
   fRnrLine   (kTRUE),
   fRnrPoints (kFALSE),
   fSmooth    (fgDefaultSmooth)
{
   fMainColorPtr = &fLineColor;
   fMarkerColor = kGreen;
}

////////////////////////////////////////////////////////////////////////////////
/// Set marker color. Propagate to projected lines.

void REveLine::SetMarkerColor(Color_t col)
{
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* l = dynamic_cast<REveLine*>(*pi);
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

void REveLine::SetLineStyle(Style_t lstyle)
{
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* pt = dynamic_cast<REveLine*>(*pi);
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

void REveLine::SetLineWidth(Width_t lwidth)
{
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* pt = dynamic_cast<REveLine*>(*pi);
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

void REveLine::SetRnrLine(Bool_t r)
{
   fRnrLine = r;
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* l = dynamic_cast<REveLine*>(*pi);
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

void REveLine::SetRnrPoints(Bool_t r)
{
   fRnrPoints = r;
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* l = dynamic_cast<REveLine*>(*pi);
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

void REveLine::SetSmooth(Bool_t r)
{
   fSmooth = r;
   std::list<REveProjected*>::iterator pi = fProjectedList.begin();
   while (pi != fProjectedList.end())
   {
      REveLine* l = dynamic_cast<REveLine*>(*pi);
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

void REveLine::ReduceSegmentLengths(Float_t max)
{
   const Float_t max2 = max*max;

   Float_t    *p = GetP();
   Int_t       s = Size();
   REveVector  a, b, d;

   std::vector<REveVector> q;

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
   for (std::vector<REveVector>::iterator i = q.begin(); i != q.end(); ++i)
      SetNextPoint(i->fX, i->fY, i->fZ);
}

////////////////////////////////////////////////////////////////////////////////
/// Sum-up lengths of individual segments.

Float_t REveLine::CalculateLineLength() const
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

REveVector REveLine::GetLineStart() const
{
   REveVector v;
   GetPoint(0, v.fX, v.fY, v.fZ);
   return v;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the last point of the line.
/// If there are no points (0,0,0) is returned.

REveVector REveLine::GetLineEnd() const
{
   REveVector v;
   GetPoint(fLastPoint, v.fX, v.fY, v.fZ);
   return v;
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.

void REveLine::CopyVizParams(const REveElement* el)
{
   const REveLine* m = dynamic_cast<const REveLine*>(el);
   if (m)
   {
      TAttLine::operator=(*m);
      fRnrLine   = m->fRnrLine;
      fRnrPoints = m->fRnrPoints;
      fSmooth    = m->fSmooth;
   }

   REvePointSet::CopyVizParams(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Write visualization parameters.

void REveLine::WriteVizParams(std::ostream& out, const TString& var)
{
   REvePointSet::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttLine::SaveLineAttributes(out, var);
   out << t << "SetRnrLine("   << ToString(fRnrLine)   << ");\n";
   out << t << "SetRnrPoints(" << ToString(fRnrPoints) << ");\n";
   out << t << "SetSmooth("    << ToString(fSmooth)    << ");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from REveProjectable, returns REvePointSetProjected class.

TClass* REveLine::ProjectedClass(const REveProjection*) const
{
   return REveLineProjected::Class();
}

////////////////////////////////////////////////////////////////////////////////
/// Get default value for smooth-line drawing flag.
/// Static function.

Bool_t REveLine::GetDefaultSmooth()
{
   return fgDefaultSmooth;
}

////////////////////////////////////////////////////////////////////////////////
/// Set default value for smooth-line drawing flag (default kFALSE).
/// Static function.

void REveLine::SetDefaultSmooth(Bool_t r)
{
   fgDefaultSmooth = r;
}

/** \class REveLineProjected
\ingroup REve
Projected copy of a REveLine.
*/


////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

REveLineProjected::REveLineProjected() :
   REveLine      (),
   REveProjected ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection manager and projection model.
/// Virtual from REveProjected.

void REveLineProjected::SetProjection(REveProjectionManager* mng,
                                      REveProjectable* model)
{
   REveProjected::SetProjection(mng, model);
   CopyVizParams(dynamic_cast<REveElement*>(model));
}

////////////////////////////////////////////////////////////////////////////////
/// Set depth (z-coordinate) of the projected points.

void REveLineProjected::SetDepthLocal(Float_t d)
{
   SetDepthCommon(d, this, fBBox);

   Int_t    n = Size();
   Float_t *p = GetP() + 2;
   for (Int_t i = 0; i < n; ++i, p+=3)
      *p = fDepth;
}

////////////////////////////////////////////////////////////////////////////////
/// Re-apply the projection.
/// Virtual from REveProjected.

void REveLineProjected::UpdateProjection()
{
   REveProjection& proj = * fManager->GetProjection();
   REveLine      & als  = * dynamic_cast<REveLine*>(fProjectable);
   REveTrans      *tr   =   als.PtrMainTrans(kFALSE);

   Int_t n = als.Size();
   Reset(n);
   fLastPoint = n - 1;
   Float_t *o = als.GetP(), *p = GetP();
   for (Int_t i = 0; i < n; ++i, o+=3, p+=3)
   {
      proj.ProjectPointfv(tr, o, p, fDepth);
   }
}
