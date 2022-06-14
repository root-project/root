// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveStraightLineSet.hxx>
#include <ROOT/REveRenderData.hxx>
#include <ROOT/REveProjectionManager.hxx>

#include "TClass.h"

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveStraightLineSet::REveStraightLineSet(const std::string& n, const std::string& t):
   REveElement (n, t),

   fLinePlex      (sizeof(Line_t), 4),
   fMarkerPlex    (sizeof(Marker_t), 8),
   fOwnLinesIds   (kFALSE),
   fOwnMarkersIds (kFALSE),
   fRnrMarkers    (kTRUE),
   fRnrLines      (kTRUE),
   fDepthTest     (kTRUE),
   fLastLine      (0)
{
   InitMainTrans();
   fPickable = kTRUE;

   fMainColorPtr = &fLineColor;
   fLineColor    = 4;
   fMarkerColor  = 2;
   fMarkerStyle  = 20;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a line.

REveStraightLineSet::Line_t*
REveStraightLineSet::AddLine(Float_t x1, Float_t y1, Float_t z1,
                             Float_t x2, Float_t y2, Float_t z2)
{
   fLastLine = new (fLinePlex.NewAtom()) Line_t(x1, y1, z1, x2, y2, z2);
   fLastLine->fId = fLinePlex.Size() - 1;
   return fLastLine;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a line.

REveStraightLineSet::Line_t*
REveStraightLineSet::AddLine(const REveVector& p1, const REveVector& p2)
{
   return AddLine(p1.fX, p1.fY, p1.fZ, p2.fX, p2.fY, p2.fZ);
}

////////////////////////////////////////////////////////////////////////////////
/// Set line vertices with given index.

void
REveStraightLineSet::SetLine(int idx,
                             Float_t x1, Float_t y1, Float_t z1,
                             Float_t x2, Float_t y2, Float_t z2)
{
   Line_t* l = (Line_t*) fLinePlex.Atom(idx);

   l->fV1[0] = x1; l->fV1[1] = y1; l->fV1[2] = z1;
   l->fV2[0] = x2; l->fV2[1] = y2; l->fV2[2] = z2;
}

////////////////////////////////////////////////////////////////////////////////
/// Set line vertices with given index.

void
REveStraightLineSet::SetLine(int idx, const REveVector& p1, const REveVector& p2)
{
   SetLine(idx, p1.fX, p1.fY, p1.fZ, p2.fX, p2.fY, p2.fZ);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a marker with given position.

REveStraightLineSet::Marker_t*
REveStraightLineSet::AddMarker(Float_t x, Float_t y, Float_t z, Int_t line_id)
{
   Marker_t* marker = new (fMarkerPlex.NewAtom()) Marker_t(x, y, z, line_id);
   return marker;
}

////////////////////////////////////////////////////////////////////////////////
/// Add a marker with given position.

REveStraightLineSet::Marker_t*
REveStraightLineSet::AddMarker(const REveVector& p, Int_t line_id)
{
   return AddMarker(p.fX, p.fY, p.fZ, line_id);
}

////////////////////////////////////////////////////////////////////////////////
/// Add a marker for line with given index on relative position pos.

REveStraightLineSet::Marker_t*
REveStraightLineSet::AddMarker(Int_t line_id, Float_t pos)
{
   Line_t& l = * (Line_t*) fLinePlex.Atom(line_id);
   return AddMarker(l.fV1[0] + (l.fV2[0] - l.fV1[0])*pos,
                    l.fV1[1] + (l.fV2[1] - l.fV1[1])*pos,
                    l.fV1[2] + (l.fV2[2] - l.fV1[2])*pos,
                    line_id);
}

////////////////////////////////////////////////////////////////////////////////
/// Copy visualization parameters from element el.

void REveStraightLineSet::CopyVizParams(const REveElement* el)
{
   const REveStraightLineSet* m = dynamic_cast<const REveStraightLineSet*>(el);
   if (m)
   {
      TAttLine::operator=(*m);
      TAttMarker::operator=(*m);
      fRnrMarkers = m->fRnrMarkers;
      fRnrLines   = m->fRnrLines;
      fDepthTest  = m->fDepthTest;
   }

   REveElement::CopyVizParams(el);
}

////////////////////////////////////////////////////////////////////////////////
/// Write visualization parameters.

void REveStraightLineSet::WriteVizParams(std::ostream& out, const TString& var)
{
   REveElement::WriteVizParams(out, var);

   TString t = "   " + var + "->";
   TAttMarker::SaveMarkerAttributes(out, var);
   TAttLine  ::SaveLineAttributes  (out, var);
   out << t << "SetRnrMarkers(" << ToString(fRnrMarkers) << ");\n";
   out << t << "SetRnrLines("   << ToString(fRnrLines)   << ");\n";
   out << t << "SetDepthTest("  << ToString(fDepthTest)  << ");\n";
}


////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveStraightLineSet::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fLinePlexSize"] = fLinePlex.Size();
   j["fMarkerPlexSize"] = fMarkerPlex.Size();
   j["fLineWidth"] = fLineWidth;
   j["fLineStyle"] = fLineStyle;
   j["fMarkerSize"] = fMarkerSize;
   j["fMarkerStyle"] = fMarkerStyle;
   j["fSecondarySelect"] = fAlwaysSecSelect;

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void REveStraightLineSet::BuildRenderData()
{
   int nVertices =  fLinePlex.Size() * 2 + fMarkerPlex.Size();
   fRenderData = std::make_unique<REveRenderData>("makeStraightLineSet", 3 * nVertices, 0, nVertices);

   // printf("REveStraightLineSet::BuildRenderData id = %d \n", GetElementId());
   REveChunkManager::iterator li(fLinePlex);
   while (li.next())
   {
      Line_t *l = (Line_t*) li();

      fRenderData->PushV(l->fV1[0], l->fV1[1],l->fV1[2]);
      fRenderData->PushV(l->fV2[0], l->fV2[1],l->fV2[2]);
      fRenderData->PushI(l->fId);
   }


   REveChunkManager::iterator mi(fMarkerPlex);
   while (mi.next())
   {
      Marker_t *m = (Marker_t*) mi();
      fRenderData->PushV(m->fV[0], m->fV[1], m->fV[2]);
      fRenderData->PushI(m->fLineId);
   }

   REveElement::BuildRenderData();

   // printf("REveStraightLineSet::BuildRenderData size= %d\n", fRenderData->GetBinarySize());
}

////////////////////////////////////////////////////////////////////////////////
/// Return class of projected object.
/// Virtual from REveProjectable.

TClass* REveStraightLineSet::ProjectedClass(const REveProjection*) const
{
   return TClass::GetClass<REveStraightLineSetProjected>();
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding-box.
/// Virtual from TAttBBox.

void REveStraightLineSet::ComputeBBox()
{
   if (fLinePlex.Size() == 0 && fMarkerPlex.Size() == 0) {
      BBoxZero();
      return;
   }

   BBoxInit();

   REveChunkManager::iterator li(fLinePlex);
   while (li.next()) {
      BBoxCheckPoint(((Line_t*)li())->fV1);
      BBoxCheckPoint(((Line_t*)li())->fV2);
   }

   REveChunkManager::iterator mi(fMarkerPlex);
   while (mi.next())
   {
      BBoxCheckPoint(((Marker_t*)mi())->fV);
   }
}


/** \class REveStraightLineSetProjected
\ingroup REve
Projected replica of a REveStraightLineSet.
*/

ClassImp(REveStraightLineSetProjected);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveStraightLineSetProjected::REveStraightLineSetProjected() :
   REveStraightLineSet(), REveProjected ()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Set projection manager and model object.

void REveStraightLineSetProjected::SetProjection(REveProjectionManager* mng,
                                                 REveProjectable* model)
{
   REveProjected::SetProjection(mng, model);

   CopyVizParams(dynamic_cast<REveElement*>(model));
}

////////////////////////////////////////////////////////////////////////////////
/// Set depth (z-coordinate) of the projected points.

void REveStraightLineSetProjected::SetDepthLocal(Float_t d)
{
   SetDepthCommon(d, this, fBBox);

   REveChunkManager::iterator li(fLinePlex);
   while (li.next())
   {
      REveStraightLineSet::Line_t& l = * (REveStraightLineSet::Line_t*) li();
      l.fV1[2] = fDepth;
      l.fV2[2] = fDepth;
   }

   REveChunkManager::iterator mi(fMarkerPlex);
   while (mi.next())
   {
      Marker_t& m = * (Marker_t*) mi();
      m.fV[2] = fDepth;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Callback that actually performs the projection.
/// Called when projection parameters have been updated.

void REveStraightLineSetProjected::UpdateProjection()
{
   REveProjection&      proj = * fManager->GetProjection();
   REveStraightLineSet& orig = * dynamic_cast<REveStraightLineSet*>(fProjectable);

   REveTrans *trans = orig.PtrMainTrans(kFALSE);

   BBoxClear();

   // Lines
   Int_t num_lines = orig.GetLinePlex().Size();
   if (proj.HasSeveralSubSpaces())
      num_lines += TMath::Max(1, num_lines/10);
   fLinePlex.Reset(sizeof(Line_t), num_lines);
   REveVector p1, p2;
   REveChunkManager::iterator li(orig.GetLinePlex());
   while (li.next())
   {
      Line_t *l = (Line_t*) li();

      proj.ProjectPointfv(trans, l->fV1, p1, fDepth);
      proj.ProjectPointfv(trans, l->fV2, p2, fDepth);

      if (proj.AcceptSegment(p1, p2, 0.1f))
      {
         AddLine(p1, p2)->fId = l->fId;
      }
      else
      {
         REveVector bp1(l->fV1), bp2(l->fV2);
         if (trans) {
            trans->MultiplyIP(bp1);
            trans->MultiplyIP(bp2);
         }
         proj.BisectBreakPoint(bp1, bp2, kTRUE, fDepth);

         AddLine(p1, bp1)->fId = l->fId;
         AddLine(bp2, p2)->fId = l->fId;
      }
   }
   if (proj.HasSeveralSubSpaces())
      fLinePlex.Refit();

   // Markers
   fMarkerPlex.Reset(sizeof(Marker_t), orig.GetMarkerPlex().Size());
   REveChunkManager::iterator mi(orig.GetMarkerPlex());
   REveVector pp;
   while (mi.next())
   {
      Marker_t &m = * (Marker_t*) mi();

      proj.ProjectPointfv(trans, m.fV, pp, fDepth);
      AddMarker(pp, m.fLineId);
   }
   StampObjProps();
}
