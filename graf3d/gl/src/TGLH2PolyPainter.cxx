#include <algorithm>
#include <stdexcept>

#include "TStopwatch.h"
#include "TStyle.h"
#include "TError.h"
#include "TColor.h"
#include "TClass.h"
#include "TMath.h"
#include "TList.h"
#include "TROOT.h"
#include "TH2Poly.h"

#include "TGLPlotCamera.h"
#include "TGLH2PolyPainter.h"
#include "TGLIncludes.h"


ClassImp(TGLH2PolyPainter)

//______________________________________________________________________________
TGLH2PolyPainter::TGLH2PolyPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord)
                   : TGLPlotPainter(hist, camera, coord, kFALSE, kFALSE, kFALSE)
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
   //Show number of bin and content, if bin is under cursor.
   fBinInfo = "";
   if (fSelectedPart) {
      if (fSelectedPart < fSelectionBase) {
         if (fHist->Class())
            fBinInfo += fHist->Class()->GetName();
         fBinInfo += "::";
         fBinInfo += fHist->GetName();
      } else if (!fHighColor) {
         const Int_t binIndex = fSelectedPart - fSelectionBase;
         TH2Poly *h = static_cast<TH2Poly *>(fHist);
         fBinInfo.Form("(bin = %d; binc = %f)", binIndex, h->GetBinContent(binIndex));
      } else
         fBinInfo = "Switch to true-color mode to obtain the correct info";
   }

   return (Char_t *)fBinInfo.Data();
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::InitGeometry()
{
   //Tesselate polygons, if not done yet.
   TH2Poly* hp = static_cast<TH2Poly *>(fHist);
   if (!fCoord->SetRanges(hp))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), 1.7,
                       fCoord->GetYRangeScaled(), 1.7,
                       fCoord->GetZRangeScaled(), 1.);

   if (hp->GetNewBinAdded()) {
      if (!CacheGeometry())
         return kFALSE;
      hp->SetNewBinAdded(kFALSE);
      hp->SetBinContentChanged(kFALSE);
   } else if (hp->GetBinContentChanged()) {
      if (!UpdateGeometry())
         return kFALSE;
      hp->SetBinContentChanged(kFALSE);
   }

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

}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawPlot()const
{
   //Draw extruded polygons and plot's frame.

   //Shift plot to point of origin.
   const Rgl::PlotTranslation trGuard(this);

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   DrawExtrusion();

//glPolygonMode(GL_FRONT, GL_LINE);

   DrawCaps();

//glPolygonMode(GL_FRONT, GL_FILL);

}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawExtrusion()const
{
   //Extruded part of bins.
   //GL_QUADS, GL_QUAD_STRIP - have the same time on my laptop, so I use
   //GL_QUADS and forgot about vertex arrays (can require more memory BTW.

   TList * bins = static_cast<TH2Poly *>(fHist)->GetBins();
   if (!bins || !bins->GetEntries()) {
      Error("TGLH2PolyPainter::DrawPlot", "Bad list of bins");
      return;
   }

   const Double_t xScale = fCoord->GetXScale();
   const Double_t yScale = fCoord->GetYScale();
   const Double_t zMin = fBackBox.Get3DBox()[0].Z();
   Double_t normal[3] = {};

   Int_t i = 0;
   for(TObjLink * link = bins->FirstLink(); link; link = link->Next(), ++i) {
      TH2PolyBin *bin = static_cast<TH2PolyBin *>(link->GetObject());
      if (!bin) {
         Error("TGLH2PolyPainter::DrawPlot", "null bin pointer in a list of bins");
         return;
      }

      const Double_t *xs = bin->GetX();
      const Double_t *ys = bin->GetY();


      if (!xs || !ys) {
         Error("TGLH2PolyPainter::DrawPlot", "null array in a bin");
         continue;
      }

      const Int_t nV = bin->GetN();
      if (nV <= 0)
         continue;

      const Double_t zMax = bin->GetContent() * fCoord->GetZScale();
      const Int_t binID = fSelectionBase + i;

      if (fSelectionPass) {
         if (!fHighColor)
            Rgl::ObjectIDToColor(binID, kFALSE);
      } else {
         SetBinColor(i);
         if(!fHighColor && fSelectedPart == binID)
            glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);
      }

      for (Int_t j = 0; j < nV - 1; ++j) {
         const Double_t v0[] = {xs[j] * xScale, ys[j] * yScale, zMin};
         const Double_t v1[] = {xs[j + 1] * xScale, ys[j + 1] * yScale, zMin};

         if (Distance(v0, v1) < 1e-10)
            continue;

         const Double_t v2[] = {v1[0], v1[1], zMax};
         const Double_t v3[] = {v0[0], v0[1], zMax};

         TMath::Normal2Plane(v0, v1, v2, normal);

         glBegin(GL_QUADS);
         glNormal3dv(normal);
         glVertex3dv(v0);
         glVertex3dv(v1);
         glVertex3dv(v2);
         glVertex3dv(v3);
         glEnd();

      }

      if (!fHighColor && !fSelectionPass && fSelectedPart == binID)
         glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);

   }
}

//______________________________________________________________________________
void TGLH2PolyPainter::DrawCaps()const
{
   //Caps on bins.
   glNormal3d(0., 0., 1.);

   typedef std::list<Rgl::Pad::Tesselation_t>::const_iterator CIter_t;
   int bin = 0;
   for (CIter_t cap = fCaps.begin(); cap != fCaps.end(); ++cap, ++bin) {
      const Int_t binID = fSelectionBase + bin;
      if (fSelectionPass) {
         if (!fHighColor)
            Rgl::ObjectIDToColor(binID, kFALSE);
      } else {
         SetBinColor(bin);
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

   const Double_t xScale = fCoord->GetXScale();
   const Double_t yScale = fCoord->GetYScale();

   fBinColors.clear();
   fBinColors.reserve(bins->GetEntries());
   fCap.clear();
   fCaps.clear();

   Rgl::Pad::Tesselator tesselator(kTRUE);

   for (TObjLink * link = bins->FirstLink(); link; link = link->Next()) {
      TH2PolyBin * b = static_cast<TH2PolyBin *>(link->GetObject());
      if (!b) {
         Error("TGH2PolyPainter::InitGeometry", "Null bin's pointer in a list of bins");
         return kFALSE;
      }

      const Double_t *xs = b->GetX();
      const Double_t *ys = b->GetY();

      if (!xs || !ys) {
         Error("TGLH2PolyPainter::DrawPlot", "null array in a bin");
         continue;
      }

      const Int_t nV = b->GetN();
      if (nV <= 0)
         continue;

      const Int_t colorIndex = gStyle->GetColorPalette(Int_t(((b->GetContent() - zMin) / (zMax - zMin)) * (nColors - 1)));
      fBinColors.push_back(colorIndex);

      const Double_t z = b->GetContent() * fCoord->GetZScale();

      fCaps.push_back(Rgl::Pad::Tesselation_t());

      fCap.resize(nV * 3);
      for (Int_t j = 0; j < nV; ++j) {
         fCap[j * 3]     = xs[j] * xScale;
         fCap[j * 3 + 1] = ys[j] * yScale;
         fCap[j * 3 + 2] = z;
      }

      tesselator.SetDump(&fCaps.back());

      GLUtesselator *t = (GLUtesselator *)tesselator.GetTess();
      gluBeginPolygon(t);
      gluNextContour(t, (GLenum)GLU_UNKNOWN);

      glNormal3d(0., 0., 1.);

      for (Int_t j = 0; j < nV; ++j)
         gluTessVertex(t, &fCap[j * 3], &fCap[j * 3]);

      gluEndPolygon(t);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLH2PolyPainter::UpdateGeometry()
{
   //Update cap's z-coordinates for all caps.
   TH2Poly *hp = static_cast<TH2Poly *>(fHist);
   TList *bins = hp->GetBins();
   if (!bins || !bins->GetEntries()) {
      Error("TGLH2PolyPainter::UpdateGeometry", "Empty list of bins in TH2Poly");
      return kFALSE;
   }

   if (Int_t(fCaps.size()) != bins->GetEntries()) {
      Error("TGLH2PolyPainter::UpdateGeometry", "Unexpected number of bins in a TH2Poly");
      return kFALSE;
   }

   std::list<Rgl::Pad::Tesselation_t>::iterator cap = fCaps.begin();
   for (TObjLink * link = bins->FirstLink(); link; link = link->Next(), ++cap) {
      TH2PolyBin * b = static_cast<TH2PolyBin *>(link->GetObject());
      if (!b) {
         Error("TGH2PolyPainter::InitGeometry", "Null bin's pointer in a list of bins");
         return kFALSE;
      }

      const Int_t nV = b->GetN();
      if (nV <= 0)
         continue;

      const Double_t z = b->GetContent() * fCoord->GetZScale();
      //Update z coordinate in all patches.
      Rgl::Pad::Tesselation_t & tess = *cap;
      Rgl::Pad::Tesselation_t::iterator patch = tess.begin();
      for (; patch != tess.end(); ++patch) {
         std::vector<Double_t> & mesh = patch->fPatch;
         for (UInt_t i = 0, e = mesh.size() / 3; i < e; ++i)
            mesh[i * 3 + 2] = z;
      }
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

namespace {

Double_t Distance(const Double_t *p1, const Double_t *p2)
{
   //Why do not we have this crap in TMath???
   //Why it it based on TGLVertex3  in TGLUtil???
   return TMath::Sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) +
                      (p1[1] - p2[1]) * (p1[1] - p2[1]) +
                      (p1[2] - p2[2]) * (p1[2] - p2[2]));
}

}
