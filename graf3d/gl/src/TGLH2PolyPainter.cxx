#include <algorithm>
#include <stdexcept>

#include "TMultiGraph.h"
#include "TH2Poly.h"
#include "TGraph.h"
#include "TClass.h"
#include "TStyle.h"
#include "TError.h"
#include "TColor.h"
#include "TMath.h"
#include "TList.h"
#include "TROOT.h"

#include "TGLH2PolyPainter.h"
#include "TGLPlotCamera.h"
#include "TGLIncludes.h"


ClassImp(TGLH2PolyPainter)

//______________________________________________________________________________
TGLH2PolyPainter::TGLH2PolyPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord)
                   : TGLPlotPainter(hist, camera, coord, kFALSE, kFALSE, kFALSE),
                     fZLog(kFALSE),
                     fZMin(0.)
{
   //Ctor.
   if(!dynamic_cast<TH2Poly *>(hist)) {
      Error("TGLH2PolyPainter::TGLH2PolyPainter", "bad histogram, must be a valid TH2Poly *");
      throw std::runtime_error("bad TH2Poly");
   }
}

//______________________________________________________________________________
char *TGLH2PolyPainter::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   //Show number of bin and bin contents, if bin is under the cursor.
   fBinInfo = "";
   if (fSelectedPart) {
      if (fSelectedPart < fSelectionBase) {
         if (fHist->Class())
            fBinInfo += fHist->Class()->GetName();
         fBinInfo += "::";
         fBinInfo += fHist->GetName();
      } else if (!fHighColor) {
         const Int_t binIndex = fSelectedPart - fSelectionBase + 1;
         TH2Poly *h = static_cast<TH2Poly *>(fHist);
         fBinInfo.Form("%s (bin = %d; binc = %f)", h->GetBinTitle(binIndex), binIndex, h->GetBinContent(binIndex));//+ 1: bins in ROOT start from 1.
      } else
         fBinInfo = "Switch to true-color mode to obtain the correct info";
   }

   return (Char_t *)fBinInfo.Data();
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::InitGeometry()
{
   //Tesselate polygons, if not done yet.
   //All pointers are validated here (and in functions called from here).
   //If any pointer is invalid - zero, or has unexpected type (dynamic_cast fails) -
   //InitGeometry will return false and nothing will be painted later.
   //That's why there are no checks in other functions.
   TH2Poly* hp = static_cast<TH2Poly *>(fHist);
   if (!fCoord->SetRanges(hp))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), Rgl::gH2PolyScaleXY,
                       fCoord->GetYRangeScaled(), Rgl::gH2PolyScaleXY,
                       fCoord->GetZRangeScaled(), 1.);

   //This code is different from lego.
   //Currently, negative bin contents are not supported.
   fZMin = fBackBox.Get3DBox()[0].Z();

   if (hp->GetNewBinAdded()) {
      if (!CacheGeometry())
         return kFALSE;
      hp->SetNewBinAdded(kFALSE);
      hp->SetBinContentChanged(kFALSE);
   } else if (hp->GetBinContentChanged() || fZLog != fCoord->GetZLog()) {
      if (!UpdateGeometry())
         return kFALSE;
      hp->SetBinContentChanged(kFALSE);
   }

   fZLog = fCoord->GetZLog();

   return kTRUE;
}

//______________________________________________________________________________
void TGLH2PolyPainter::StartPan(Int_t px, Int_t py)
{
   //User clicks on a lego with middle mouse button (middle for pad).
   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, py);
}

//______________________________________________________________________________
void TGLH2PolyPainter::Pan(Int_t px, Int_t py)
{
      //Mouse events handler.
   if (fSelectedPart >= fSelectionBase) {//Pan camera.
      SaveModelviewMatrix();
      SaveProjectionMatrix();

      fCamera->SetCamera();
      fCamera->Apply(fPadPhi, fPadTheta);
      fCamera->Pan(px, py);

      RestoreProjectionMatrix();
      RestoreModelviewMatrix();
   } else if (fSelectedPart > 0) {
      //Convert py into bottom-top orientation.
      py = fCamera->GetHeight() - py;

      SaveModelviewMatrix();
      SaveProjectionMatrix();

      fCamera->SetCamera();
      fCamera->Apply(fPadPhi, fPadTheta);

      if (!fHighColor) {
         if (fBoxCut.IsActive() && (fSelectedPart >= kXAxis && fSelectedPart <= kZAxis)) {
            fBoxCut.MoveBox(px, py, fSelectedPart);
         }
      }

      RestoreProjectionMatrix();
      RestoreModelviewMatrix();
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLH2PolyPainter::AddOption(const TString &/*stringOption*/)
{
   //No additional options.
}

//______________________________________________________________________________
void TGLH2PolyPainter::ProcessEvent(Int_t /*event*/, Int_t /*px*/, Int_t /*py*/)
{
   //No events.
}

//______________________________________________________________________________
void TGLH2PolyPainter::InitGL()const
{
   //Initialize some gl state variables.
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGLH2PolyPainter::DeInitGL()const
{
   //Return some gl states to original values.
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glDisable(GL_LIGHT0);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

namespace {

Double_t Distance(const Double_t *p1, const Double_t *p2);
Bool_t   IsPolygonCW(const Double_t *xs, const Double_t *ys, Int_t n);

}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawPlot()const
{
   //Draw extruded polygons and plot's frame.

   //Shift plot to point of origin.
   const Rgl::PlotTranslation trGuard(this);

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   DrawExtrusion();
   DrawCaps();
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawExtrusion()const
{
   //Extruded part of bins.
   //GL_QUADS, GL_QUAD_STRIP - have the same time on my laptop, so I use
   //GL_QUADS and forgot about vertex arrays (can require more memory BTW).
   TList *bins = static_cast<TH2Poly *>(fHist)->GetBins();
   Int_t binIndex = 0;
   for(TObjLink * link = bins->FirstLink(); link; link = link->Next(), ++binIndex) {
      TH2PolyBin *bin = static_cast<TH2PolyBin *>(link->GetObject());
      //const Double_t zMax = bin->GetContent() * fCoord->GetZScale();
      Double_t zMax = bin->GetContent();
      ClampZ(zMax);

      if (const TGraph * poly = dynamic_cast<TGraph*>(bin->GetPolygon())) {
         //DrawExtrusion(poly, zMin, zMax, binIndex);
         DrawExtrusion(poly, fZMin, zMax, binIndex);
      } else if (const TMultiGraph * mg = dynamic_cast<TMultiGraph*>(bin->GetPolygon())) {
         //DrawExtrusion(mg, zMin, zMax, binIndex);
         DrawExtrusion(mg, fZMin, zMax, binIndex);
      }//else is impossible.
   }
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawExtrusion(const TGraph *poly, Double_t zMin, Double_t zMax, Int_t binIndex)const
{
   //Extrude polygon, described by TGraph.
   const Double_t *xs = poly->GetX();
   const Double_t *ys = poly->GetY();
       
   const Int_t nV = poly->GetN();

   //nV can never be less than 3 - InitGeometry fails on such polygons.
   //So, no checks here.

   const Int_t binID = fSelectionBase + binIndex;

   if (fSelectionPass) {
      if (!fHighColor)
         Rgl::ObjectIDToColor(binID, kFALSE);
   } else {
      SetBinColor(binIndex);
      if(!fHighColor && fSelectedPart == binID)
         glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);
   }

   //Before, orientation was always done in TH2Poly.
   //Now we do it every time here.
   FillTemporaryPolygon(xs, ys, 0., nV); //0. == z is not important here.

   Double_t normal[3] = {};
   for (Int_t j = 0; j < nV - 1; ++j) {
      const Double_t v0[] = {fPolygon[j * 3], fPolygon[j * 3 + 1], zMin};
      const Double_t v1[] = {fPolygon[(j + 1) * 3], fPolygon[(j + 1) * 3 + 1], zMin};

      if (Distance(v0, v1) < 1e-10)
         continue;

      const Double_t v2[] = {v1[0], v1[1], zMax};
      const Double_t v3[] = {v0[0], v0[1], zMax};

      TMath::Normal2Plane(v0, v1, v2, normal);
      Rgl::DrawQuadFilled(v0, v1, v2, v3, normal);
   }

   //Now, close the polygon.
   const Double_t v0[] = {fPolygon[(nV - 1) * 3], fPolygon[(nV - 1) * 3 + 1], zMin};
   const Double_t v1[] = {fPolygon[0], fPolygon[1], zMin};

   if (Distance(v0, v1) > 1e-10) {
      const Double_t v2[] = {v1[0], v1[1], zMax};
      const Double_t v3[] = {v0[0], v0[1], zMax};

      TMath::Normal2Plane(v0, v1, v2, normal);
      Rgl::DrawQuadFilled(v0, v1, v2, v3, normal);
   }

   if (!fHighColor && !fSelectionPass && fSelectedPart == binID)
      glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawExtrusion(const TMultiGraph *mg, Double_t zMin, Double_t zMax, Int_t binIndex)const
{
   //Multigraph contains a list of graphs, draw them.
   const TList *graphs = mg->GetListOfGraphs();
   for (TObjLink *link = graphs->FirstLink(); link; link = link->Next())
      DrawExtrusion((TGraph *)(link->GetObject()), zMin, zMax, binIndex);
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawCaps()const
{
   //Caps on bins.
   glNormal3d(0., 0., 1.);

   Int_t binIndex = 0;
   const TList *bins = static_cast<TH2Poly *>(fHist)->GetBins();
   CIter_t cap = fCaps.begin();

   if(!bins->FirstLink())
      throw 1;

   //Very ugly iteration statement. Number of caps is equal to number of links (in a list
   //of bins including nested links in multigraphs. But this is not obvious from the code here,
   //so, I've added check cap != fCaps.end() to make it more or less clear.
   for (TObjLink *link = bins->FirstLink(); link && cap != fCaps.end(); link = link->Next()) {
      TH2PolyBin *polyBin = static_cast<TH2PolyBin *>(link->GetObject());
      if (dynamic_cast<TGraph *>(polyBin->GetPolygon())) {
         DrawCap(cap, binIndex);
         ++cap;
      } else if (TMultiGraph *mg = dynamic_cast<TMultiGraph *>(polyBin->GetPolygon())) {
         const TList *gs = mg->GetListOfGraphs();
         TObjLink *graphLink = gs->FirstLink();
         for (; graphLink && cap != fCaps.end(); graphLink = graphLink->Next(), ++cap)
            DrawCap(cap, binIndex);
      }

      ++binIndex;
   }
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawCap(CIter_t cap, Int_t binIndex)const
{
   //Draw a cap on top of a bin.
   const Int_t binID = fSelectionBase + binIndex;
   if (fSelectionPass) {
      if (!fHighColor)
         Rgl::ObjectIDToColor(binID, kFALSE);
   } else {
      SetBinColor(binIndex);
      if(!fHighColor && fSelectedPart == binID)
         glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);
   }

   const Rgl::Pad::Tesselation_t &t = *cap;
   typedef std::list<Rgl::Pad::MeshPatch_t>::const_iterator CMIter_t;
   for (CMIter_t p = t.begin(); p != t.end(); ++p) {
      const std::vector<Double_t> &vs = p->fPatch;
      glBegin(GLenum(p->fPatchType));
      for (UInt_t i = 0; i < vs.size(); i += 3)
         glVertex3dv(&vs[i]);
      glEnd();
   }

   if (!fHighColor && !fSelectionPass && fSelectedPart == binID)
      glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::CacheGeometry()
{
   //Cache all data for TH2Poly object.
   TH2Poly *hp = static_cast<TH2Poly *>(fHist);
   TList *bins = hp->GetBins();
   if (!bins || !bins->GetEntries()) {
      Error("TGLH2PolyPainter::CacheGeometry", "Empty list of bins in TH2Poly");
      return kFALSE;
   }

   const Double_t zMin = fHist->GetMinimum();
   const Double_t zMax = fHist->GetMaximum();
   const Int_t nColors = gStyle->GetNumberOfColors();

   fBinColors.clear();
   fBinColors.reserve(bins->GetEntries());
   fPolygon.clear();
   fCaps.clear();

   Rgl::Pad::Tesselator tesselator(kTRUE);

   for (TObjLink * link = bins->FirstLink(); link; link = link->Next()) {
      TH2PolyBin * bin = static_cast<TH2PolyBin *>(link->GetObject());
      if (!bin || !bin->GetPolygon()) {
         Error("TGH2PolyPainter::InitGeometry", "Null bin or polygon pointer in a list of bins");
         return kFALSE;
      }

      Double_t binZ = bin->GetContent();
      if (!ClampZ(binZ)) {
         Error("TGLH2PolyPainter::CacheGeometry", "Negative bin content and log scale");
         return kFALSE;
      }

      if (const TGraph *g = dynamic_cast<TGraph *>(bin->GetPolygon())) {
         if (!BuildTesselation(tesselator, g,  binZ))
            return kFALSE;
      } else if (const TMultiGraph *mg = dynamic_cast<TMultiGraph *>(bin->GetPolygon())) {
         if (!BuildTesselation(tesselator, mg, binZ))
            return kFALSE;
      } else {
         //Da vy chto, sgovorilis' chto li???
         Error("TGLH2PolyPainter::CacheGeometry", "Bin contains object of unknown type");
         return kFALSE;
      }

      const Int_t colorIndex = gStyle->GetColorPalette(Int_t(((bin->GetContent() - zMin) / (zMax - zMin)) * (nColors - 1)));
      fBinColors.push_back(colorIndex);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::BuildTesselation(Rgl::Pad::Tesselator &tess, const TGraph *g, Double_t z)
{
   //Tesselate a polygon described by TGraph.
   const Double_t *xs = g->GetX();
   const Double_t *ys = g->GetY();

   if (!xs || !ys) {
      Error("TGLH2PolyPainter::BuildTesselation", "null array(s) in a polygon");
      return kFALSE;
   }

   const Int_t nV = g->GetN();
   if (nV < 3) {
      Error("TGLH2PolyPainter::BuildTesselation", "number of vertices in a polygon must be >= 3");
      return kFALSE;
   }

   fCaps.push_back(Rgl::Pad::Tesselation_t());
   FillTemporaryPolygon(xs, ys, z, nV);

   tess.SetDump(&fCaps.back());

   GLUtesselator *t = (GLUtesselator *)tess.GetTess();
   gluBeginPolygon(t);
   gluNextContour(t, (GLenum)GLU_UNKNOWN);

   glNormal3d(0., 0., 1.);

   for (Int_t j = 0; j < nV; ++j) {
      gluTessVertex(t, &fPolygon[j * 3], &fPolygon[j * 3]);
   }
   gluEndPolygon(t);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::BuildTesselation(Rgl::Pad::Tesselator &tess, const TMultiGraph *mg, Double_t z)
{
   //Iterate over multi graph contents and tesselate nested TGraphs.
   const TList *graphs = mg->GetListOfGraphs();
   if (!graphs) {
      Error("TGLH2PolyPainter::BuildTesselation", "null list of graphs in a multigraph");
      return kFALSE;
   }

   for(TObjLink *link = graphs->FirstLink(); link; link = link->Next()) {
      const TGraph *graph = dynamic_cast<TGraph *>(link->GetObject());
      if (!graph) {
         //Da chto za fignia v konce koncov????
         Error("TGLH2PolyPainter::BuildTesselation", "TGraph expected inside a multigraph, got something else");
         return kFALSE;
      }

      if (!BuildTesselation(tess, graph, z))
         return kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::UpdateGeometry()
{
   //Update cap's z-coordinates for all caps.
   //Here no pointers are checked, this was already done
   //by InitGeometry. So, if histogram was broken somehow
   //- hehe, good luck.
   TH2Poly *hp = static_cast<TH2Poly *>(fHist);
   TList *bins = hp->GetBins();

   std::list<Rgl::Pad::Tesselation_t>::iterator cap = fCaps.begin();

   //Ugly iteration statements, but still, number of links and caps will be
   //ok - this is checked in InitGeometry. Check cap != fCaps.end() is added
   //to make it clear (not required).
   for (TObjLink *link = bins->FirstLink(); link && cap != fCaps.end(); link = link->Next()) {
      TH2PolyBin *b = static_cast<TH2PolyBin *>(link->GetObject());
      Double_t z = b->GetContent();
      ClampZ(z);
      //Update z coordinate in all patches.
      if (dynamic_cast<TGraph *>(b->GetPolygon())) {
         //Only one cap.
         Rgl::Pad::Tesselation_t &tess = *cap;
         Rgl::Pad::Tesselation_t::iterator patch = tess.begin();
         for (; patch != tess.end(); ++patch) {
            std::vector<Double_t> &mesh = patch->fPatch;
            for (UInt_t i = 0, e = mesh.size() / 3; i < e; ++i)
               mesh[i * 3 + 2] = z;
         }

         ++cap;
      } else if (const TMultiGraph *mg = dynamic_cast<TMultiGraph *>(b->GetPolygon())) {
         const TList *gs = mg->GetListOfGraphs();
         for (TObjLink * graphLink = gs->FirstLink(); graphLink && cap != fCaps.end(); graphLink = graphLink->Next(), ++cap) {
            Rgl::Pad::Tesselation_t &tess = *cap;
            Rgl::Pad::Tesselation_t::iterator patch = tess.begin();
            for (; patch != tess.end(); ++patch) {
               std::vector<Double_t> &mesh = patch->fPatch;
               for (UInt_t i = 0, e = mesh.size() / 3; i < e; ++i)
                  mesh[i * 3 + 2] = z;
            }
         }
      }//else is impossible and processed by InitGeometry.
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGLH2PolyPainter::SetBinColor(Int_t binIndex)const
{
   //Set bin's color.
   if (binIndex >= Int_t(fBinColors.size())) {
      Error("TGLH2PolyPainter::SetBinColor", "bin index is out of range %d, must be <= %d",
            binIndex, int(fBinColors.size()));
      return;
   }

   //Convert color index into RGB.
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.15f};

   if (const TColor *c = gROOT->GetColor(fBinColors[binIndex]))
      c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawSectionXOZ()const
{
   //No sections.
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawSectionYOZ()const
{
   //No sections.
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawSectionXOY()const
{
   //No sections.
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawPalette()const
{
   //Not yet.
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawPaletteAxis()const
{
   //Not yet.
}

//______________________________________________________________________________
void TGLH2PolyPainter::FillTemporaryPolygon(const Double_t *xs, const Double_t *ys, Double_t z, Int_t nV)const
{
   //Since I probably have to re-orient polygon, I need a temporary polygon.
   const Double_t xScale = fCoord->GetXScale();
   const Double_t yScale = fCoord->GetYScale();
  
   fPolygon.resize(nV * 3);
   for (Int_t j = 0; j < nV; ++j) {
      fPolygon[j * 3]     = xs[j] * xScale;
      fPolygon[j * 3 + 1] = ys[j] * yScale;
      fPolygon[j * 3 + 2] = z;
   }

   if (IsPolygonCW(xs, ys, nV))
      MakePolygonCCW();
}

//______________________________________________________________________________
void TGLH2PolyPainter::MakePolygonCCW()const
{
   //Code taken from the original TH2Poly. 
   const Int_t nV = Int_t(fPolygon.size() / 3);
   for (Int_t a = 0; a <= (nV / 2) - 1; a++) {
      const Int_t b = nV - 1 - a;
      std::swap(fPolygon[a * 3], fPolygon[b * 3]);
      std::swap(fPolygon[a * 3 + 1], fPolygon[b * 3 + 1]);
   }
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::ClampZ(Double_t &zVal)const
{
   //Clamp z value.
   if (fCoord->GetZLog()) {
      if (zVal <= 0.)
         return kFALSE;
      else
         zVal = TMath::Log10(zVal) * fCoord->GetZScale();
   } else
      zVal *= fCoord->GetZScale();

   const TGLVertex3 *frame = fBackBox.Get3DBox();

   if (zVal > frame[4].Z())
      zVal = frame[4].Z();
   else if (zVal < frame[0].Z())
      zVal = frame[0].Z();

   return kTRUE;
}


namespace {

//______________________________________________________________________________
Double_t Distance(const Double_t *p1, const Double_t *p2)
{
   //
   return TMath::Sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                      (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                      (p1[2] - p2[2]) * (p1[2] - p2[2]));
}

//______________________________________________________________________________
Bool_t IsPolygonCW(const Double_t *xs, const Double_t *ys, Int_t n)
{
   //Before, TH2Poly always produced good GL polygons - CCW. Now,
   //this code (by Deniz Gunceler) was deleted from TH2Poly.
   Double_t signedArea = 0.;

   for (Int_t j = 0; j < n - 1; ++j)
      signedArea += xs[j] * ys[j + 1] - ys[j] * xs[j + 1];

   return signedArea < 0.;         
}

}
