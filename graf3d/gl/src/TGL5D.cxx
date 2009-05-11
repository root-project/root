// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <algorithm>
#include <stdexcept>
#include <typeinfo>

#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "TVirtualX.h"
#include "Buttons.h"
#include "TString.h"
#include "TError.h"
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"
#include "TH3.h"

#include "TGLMarchingCubes.h"
#include "TGLPlotCamera.h"
#include "TGLPadUtils.h"
#include "TGLIncludes.h"
#include "TGL5D.h"



namespace {

const double gEps = 1e-3;

/*
Auxilary functions to draw iso-meshes.
Now simply duplicates TGLTF3Painter.cxx contents.
Must be moved to TGLUtils or TGLPlotPainter to avoid
code duplication.
*/

//______________________________________________________________________________
template<class V>
void DrawMesh(GLenum type, const std::vector<V> &vs, const std::vector<V> &ns, 
              const std::vector<UInt_t> &fTS)
{
   //Surface with material and lighting.
   glEnableClientState(GL_VERTEX_ARRAY);
   glEnableClientState(GL_NORMAL_ARRAY);
   glVertexPointer(3, type, 0, &vs[0]);
   glNormalPointer(type, 0, &ns[0]);
   glDrawElements(GL_TRIANGLES, fTS.size(), GL_UNSIGNED_INT, &fTS[0]);
   glDisableClientState(GL_NORMAL_ARRAY);
   glDisableClientState(GL_VERTEX_ARRAY);
}

//______________________________________________________________________________
template<class V>
void DrawMesh(GLenum type, const std::vector<V> &vs, const std::vector<UInt_t> &fTS)
{
   //Only vertices, no normal (no lighting and material).
   glEnableClientState(GL_VERTEX_ARRAY);
   glVertexPointer(3, type, 0, &vs[0]);
   glDrawElements(GL_TRIANGLES, fTS.size(), GL_UNSIGNED_INT, &fTS[0]);
   glDisableClientState(GL_VERTEX_ARRAY);
}

//______________________________________________________________________________
template<class V, class GLN, class GLV>
void DrawMesh(GLN normal3, GLV vertex3, const std::vector<V> &vs, 
              const std::vector<V> &ns, const std::vector<UInt_t> &fTS, 
              const TGLBoxCut &box)
{
   //Mesh with cut.
   //Material and lighting are enabled.
   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t * t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;

      normal3(&ns[t[0] * 3]);
      vertex3(&vs[t[0] * 3]);
      
      normal3(&ns[t[1] * 3]);
      vertex3(&vs[t[1] * 3]);
      
      normal3(&ns[t[2] * 3]);
      vertex3(&vs[t[2] * 3]);
   }

   glEnd();
}

//______________________________________________________________________________
template<class V, class GLV>
void DrawMesh(GLV vertex3, const std::vector<V> &vs, const std::vector<UInt_t> &fTS, 
              const TGLBoxCut &box)
{
   //Mesh with cut.
   //No material and lighting.
   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t * t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;

      vertex3(&vs[t[0] * 3]);
      vertex3(&vs[t[1] * 3]);
      vertex3(&vs[t[2] * 3]);
   }

   glEnd();
}

//______________________________________________________________________________
void GetColor(Double_t *rfColor, const Double_t *n)
{
   //GetColor generates a color from a given normal
   const Double_t x = n[0];
   const Double_t y = n[1];
   const Double_t z = n[2];
   rfColor[0] = (x > 0. ? x : 0.) + (y < 0. ? -0.5 * y : 0.) + (z < 0. ? -0.5 * z : 0.);
   rfColor[1] = (y > 0. ? y : 0.) + (z < 0. ? -0.5 * z : 0.) + (x < 0. ? -0.5 * x : 0.);
   rfColor[2] = (z > 0. ? z : 0.) + (x < 0. ? -0.5 * x : 0.) + (y < 0. ? -0.5 * y : 0.);
}

//______________________________________________________________________________
void DrawMapleMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
                   const std::vector<UInt_t> &fTS)
{
   //Colored mesh with lighting disabled.
   Double_t color[] = {0., 0., 0., 0.15};

   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t *t = &fTS[i * 3];
      const Double_t * n = &ns[t[0] * 3];
      //
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[0] * 3]);
      //
      n = &ns[t[1] * 3];
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[1] * 3]);
      //
      n = &ns[t[2] * 3];
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[2] * 3]);
   }

   glEnd();
}

void DrawMapleMesh(const std::vector<Double_t> &vs, const std::vector<Double_t> &ns,
                   const std::vector<UInt_t> &fTS, const TGLBoxCut & box)
{
   //Colored mesh with cut and disabled lighting.
   Double_t color[] = {0., 0., 0., 0.15};

   glBegin(GL_TRIANGLES);

   for (UInt_t i = 0, e = fTS.size() / 3; i < e; ++i) {
      const UInt_t *t = &fTS[i * 3];
      if (box.IsInCut(&vs[t[0] * 3]))
         continue;
      if (box.IsInCut(&vs[t[1] * 3]))
         continue;
      if (box.IsInCut(&vs[t[2] * 3]))
         continue;
      const Double_t * n = &ns[t[0] * 3];
      //
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[0] * 3]);
      //
      n = &ns[t[1] * 3];
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[1] * 3]);
      //
      n = &ns[t[2] * 3];
      GetColor(color, n);
      glColor4dv(color);
      glVertex3dv(&vs[t[2] * 3]);
   }

   glEnd();
}

}//Unnamed namespace.

//______________________________________________________________________________
//
// "gl5d" option for TH3F.

ClassImp(TGL5D)

//______________________________________________________________________________
TGL5D::TGL5D(TH3F *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord)
         : TGLPlotPainter(hist, camera, coord, kFALSE, kFALSE, kFALSE),
           fInit(kFALSE),
           /*fV5SliderMin(0.), fV5SliderMax(0.),*/
           fNP(0),
           fV1(0), fV2(0), fV3(0), fV4(0), fV5(0),
           fShowSlider(kFALSE)
{
   //Constructor.
}

namespace {
void FindRange(Int_t size, const Double_t *src, Rgl::Range_t &range);
}

//______________________________________________________________________________
void TGL5D::SetDataSources(Int_t size, const Double_t *v1, const Double_t *v2, const Double_t *v3,
                           const Double_t *v4, const Double_t *v5)
{
   if (!size || !v1 || !v2 || !v3 || !v4 || !v5) {
      Error("TGL5D::SetDataSources", "Set data sources correctly.");
      return;
   }
   //Copy data from external vectors.
   fNP = size;
   fV1 = v1;
   fV2 = v2;
   fV3 = v3;
   fV4 = v4;
   fV5 = v5;
   FindRange(size, v1, fV1MinMax);
   FindRange(size, v2, fV2MinMax);
   FindRange(size, v3, fV3MinMax);
   FindRange(size, v4, fV4MinMax);
   FindRange(size, v5, fV5MinMax);
   /*
   fV5SliderMin = fV5MinMax.first;
   fV5SliderMax = fV5MinMax.second;*/
}

//______________________________________________________________________________
const Rgl::Range_t &TGL5D::GetV1Range()const
{
   //Range for the first variable.
   return fV1MinMax;
}

//______________________________________________________________________________
const Rgl::Range_t &TGL5D::GetV2Range()const
{
   //Range for the second variable.
   return fV2MinMax;
}

//______________________________________________________________________________
const Rgl::Range_t &TGL5D::GetV3Range()const
{
   //Range for the third variable.
   return fV3MinMax;
}

//______________________________________________________________________________
const Rgl::Range_t &TGL5D::GetV4Range()const
{
   //Range for the forth variable.
   return fV4MinMax;
}

//______________________________________________________________________________
const Rgl::Range_t &TGL5D::GetV5Range()const
{
   //Range for the fith variable.
   return fV5MinMax;
}
/*
//______________________________________________________________________________
void TGL5D::SetV5SliderMin(Double_t min)
{
   if (min > fV5MinMax.second || min < fV5MinMax.first)
      return;
   fV5SliderMin = min;
}

//______________________________________________________________________________
void TGL5D::SetV5SliderMax(Double_t max)
{
   if (max > fV5MinMax.second || max < fV5MinMax.first)
      return;
   fV5SliderMax = max;
}
*/
namespace {
//This must be calculated by regression tool.
/*
//______________________________________________________________________________
Double_t Emulate5th(const Float_t *v)
{
   return v[0] * v[0] - v[1] * v[1] * 0.3 + v[2] * 5.;
}
*/

}

//______________________________________________________________________________
TGL5D::SurfIter_t TGL5D::AddSurface(Double_t v4, Color_t ci, Double_t iso, Double_t sigma, 
                             Double_t eVal, Double_t range, Int_t lownps)
{
   //Try to add new iso-surface.
   //If something goes wrong, return
   //pointer to the end of fIsos - so, such
   //iterator can be checked later in TGL5D
   //functions. Do not use this iterator externally.
   if (!fNP || !fV1 || !fV2 || !fV3 || !fV4 || !fV5) {
      Error("TGL5D::AddSurface", "Set data sources correctly before adding a surface.");
      return fIsos.end();
   }
   
   fPtsSorted.clear();

   for (Int_t i = 0; i < fNP; ++i) {
      if (TMath::Abs(fV4[i] - v4) < range) {
         fPtsSorted.push_back(fV1[i]);//x
         fPtsSorted.push_back(fV2[i]);//y
         fPtsSorted.push_back(fV3[i]);//z
      }
   }
   
   if (fPtsSorted.size() / 3 < size_type(lownps)) {
      Warning("TGL5D::AddNewSurface", "Number of selected points is too small: %d", Int_t(fPtsSorted.size()));
      return fIsos.end();//This is valid iterator, but invalid surface.
   }

   fKDE.BuildModel(fPtsSorted, sigma, 3, 8);

   const UInt_t nZ = fZAxis->GetNbins();
   const UInt_t nY = fYAxis->GetNbins();
   const UInt_t nX = fXAxis->GetNbins();

   Info("TGL5D::AddSurface", "Preparing targets ...");

   fTS.clear();
   fTS.reserve(nX * nY * nZ * 3);
   
   for (UInt_t i = 1; i <= nZ; ++i) {
      for(UInt_t j = 1; j <= nY; ++j) {
         for(UInt_t k = 1; k <= nX; ++k) {
            fTS.push_back(fXAxis->GetBinCenter(k));
            fTS.push_back(fYAxis->GetBinCenter(j));
            fTS.push_back(fZAxis->GetBinCenter(i));
         }
      }
   }
   
   Info("TGL5D::AddSurface", "Targets are ready.");
   fDens.assign(fTS.size() / 3, 0);
   
   fKDE.Predict(fTS, fDens, eVal);
   //Now we have densities on a regular grid and can build a mesh.
   for(UInt_t i = 1, ind = 0; i <= nZ; ++i)
      for(UInt_t j = 1; j <= nY; ++j)
         for(UInt_t k = 1; k <= nX; ++k)
            fHist->SetBinContent(k, j, i, fDens[ind++]);


   Info("TGL5D::AddSurface", "Building the mesh ...");
   Rgl::Mc::TGridGeometry<Float_t> geom;
   //Get grid parameters.
   geom.fMinX  = fXAxis->GetBinCenter(fXAxis->GetFirst());
   geom.fStepX = (fXAxis->GetBinCenter(fXAxis->GetLast()) - geom.fMinX) / (fHist->GetNbinsX() - 1);
   geom.fMinY  = fYAxis->GetBinCenter(fYAxis->GetFirst());
   geom.fStepY = (fYAxis->GetBinCenter(fYAxis->GetLast()) - geom.fMinY) / (fHist->GetNbinsY() - 1);
   geom.fMinZ  = fZAxis->GetBinCenter(fZAxis->GetFirst());
   geom.fStepZ = (fZAxis->GetBinCenter(fZAxis->GetLast()) - geom.fMinZ) / (fHist->GetNbinsZ() - 1);
   //Scale grid parameters.
   geom.fMinX *= fCoord->GetXScale(), geom.fStepX *= fCoord->GetXScale();
   geom.fMinY *= fCoord->GetYScale(), geom.fStepY *= fCoord->GetYScale();
   geom.fMinZ *= fCoord->GetZScale(), geom.fStepZ *= fCoord->GetZScale();
   
   Mesh_t mesh;
   Rgl::Mc::TMeshBuilder<TH3F, Float_t> builder(kTRUE);
   builder.BuildMesh(static_cast<TH3F *>(fHist), geom, &mesh, iso);

   Info("TGL5D::AddSurface", "Mesh has %d vertices", Int_t(mesh.fVerts.size() / 3));
   
   if (!mesh.fVerts.size())//I do not need an empty mesh.
      return fIsos.end();
   //Add surface with empty mesh and swap meshes.
   fIsos.push_front(fDummy);
   
   fIsos.front().fMesh.Swap(mesh);
   fIsos.front().f4D = v4;
   fIsos.front().fRange = range;
   fIsos.front().fShowCloud = kFALSE;
   fIsos.front().fHide = kFALSE;
   fIsos.front().fColor = ci;
  /* 
   //This part is only for 5d demo.
   std::vector<Float_t>  &m = fIsos.front().fMesh.fVerts;
   std::vector<Double_t> &p = fIsos.front().fPreds;
   
   p.assign(m.size() / 3, 0.);
   p[0] = Emulate5th(&m[0]);
   
   if (fIsos.size()) {
      fV5MinMax.first = TMath::Min(fV5MinMax.first, p[0]);
      fV5MinMax.second = TMath::Max(fV5MinMax.second, p[0]);
   } else
      fV5MinMax.second = fV5MinMax.first = p[0];
      
   for (size_type i = 1, e = m.size() / 3; i < e; ++i) {
      const Double_t val = Emulate5th(&m[i * 3]);
      fV5MinMax.first = TMath::Min(fV5MinMax.first, val);
      fV5MinMax.second = TMath::Max(fV5MinMax.second, val);
      p[i] = val;
   }
   
   fV5SliderMin = fV5MinMax.first;
   fV5SliderMax = 0.01 * (fV5MinMax.second - fV5MinMax.first);
   */
   return fIsos.begin();
}

//______________________________________________________________________________
void TGL5D::SetSurfaceMode(SurfIter_t surf, Bool_t cloudOn)
{
   //Show/hide cloud for surface.
   if (surf == fIsos.end()) {
      Error("TGL5D::SetSurfaceMode", "Invalid iterator, no such surface exists.");
      return;
   }
   
   if (surf->fShowCloud != cloudOn) {
      surf->fShowCloud = cloudOn;
      gPad->Update();
   }
}

//______________________________________________________________________________
void TGL5D::SetSurfaceColor(SurfIter_t surf, Color_t ci)
{
   //Change the color for iso-surface.
   if (surf == fIsos.end()) {
      Error("TGL5D::SetSurfaceColor", "Invalid iterator, no such surface exists.");
      return;
   }
   
   surf->fColor = ci;
   gPad->Update();
}

//______________________________________________________________________________
void TGL5D::HideSurface(SurfIter_t surf)
{
   //Hide iso-surface.
   if (surf == fIsos.end()) {
      Error("TGL5D::HideSurface", "Invalid iterator, no such surface exists.");
      return;
   }
   
   surf->fHide = kTRUE;
   gPad->Update();
}

//______________________________________________________________________________
void TGL5D::ShowSurface(SurfIter_t surf)
{
   //Show previously hidden iso-surface.
   if (surf == fIsos.end()) {
      Error("TGL5D::ShowSurface", "Invalid iterator, no such surface exists.");
      return;
   }
   
   surf->fHide = kFALSE;
   gPad->Update();
}

//______________________________________________________________________________
void TGL5D::RemoveSurface(SurfIter_t surf)
{
   //Remove iso-surface.
   if (surf == fIsos.end()) {
      Error("TGL5D::RemoveSurface", "Invalid iterator, no such surface exists.");
      return;
   }

   fIsos.erase(surf);
   gPad->Update();
}

//______________________________________________________________________________
char *TGL5D::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   //Return info for plot part under cursor.
   static char mess[] = { "gl5d" };
   return mess;
}

//______________________________________________________________________________
Bool_t TGL5D::InitGeometry()
{
   //Create mesh.
   if (fInit)
      return kTRUE;

   //Only in cartesian.
   fCoord->SetCoordType(kGLCartesian);
   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   if (fCamera) fCamera->SetViewVolume(fBackBox.Get3DBox());

   return kTRUE;
}

//______________________________________________________________________________
void TGL5D::StartPan(Int_t px, Int_t py)
{
   //User clicks right mouse button (in a pad).
   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}

//______________________________________________________________________________
void TGL5D::Pan(Int_t px, Int_t py)
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
void TGL5D::AddOption(const TString &/*option*/)
{
   //No additional options for TGL5D.
}

//______________________________________________________________________________
void TGL5D::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{

   //Change color sheme.
   if (event == kKeyPress) {
      if (py == kKey_c || py == kKey_C) {
         if (fHighColor)
            Info("ProcessEvent", "Cut box does not work in high color, please, switch to true color");
         else {
            fBoxCut.TurnOnOff();
            fUpdateSelection = kTRUE;
         }
      } else if (py == kKey_r || py == kKey_R) {
/*         const Double_t range = fV5SliderMax - fV5SliderMin;
         const Double_t delta = 0.2 * range;
         if (fV5SliderMax + delta < fV5MinMax.second)
            fV5SliderMin += delta, fV5SliderMax += delta;
         else
            fV5SliderMin = fV5MinMax.first, fV5SliderMax = fV5SliderMin + range;*/
      }
   } else if (event == kButton1Double && fBoxCut.IsActive()) {
      if (fBoxCut.IsActive())
         fBoxCut.TurnOnOff();
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLPlotPainter *)0x%lx)->Paint()", this));
      else
         Paint();
   }
}

//______________________________________________________________________________
void TGL5D::InitGL() const
{
   //Initialize OpenGL state variables.
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGL5D::DeInitGL()const
{
   //Return some gl states to original values.
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
   glDisable(GL_CULL_FACE);   
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHT0);
   glDisable(GL_LIGHTING);
}

//______________________________________________________________________________
void TGL5D::DrawPlot() const
{
   //Draw set of meshes.
   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);
   //
   if (!fIsos.size())
      DrawCloud();
   else {
      for (ConstSurfIter_t it = fIsos.begin(); it != fIsos.end(); ++it) {
         if (!fSelectionPass)
            SetSurfaceColor(it->fColor);
         DrawMesh(it);
         if (it->fShowCloud && !fSelectionPass)
            DrawSubCloud(it->f4D, it->fRange, it->fColor);
      }      
   }
   
   if (fBoxCut.IsActive())
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);
}

//______________________________________________________________________________
void TGL5D::SetSurfaceColor(Color_t ind)const
{
   //Set the color for iso-surface.
   Float_t rgba[] = {0.f, 0.f, 0.f, 0.5f};
   Rgl::Pad::ExtractRGB(ind, rgba);
   //Set color for surface.
   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, rgba);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 20.f);
}

//______________________________________________________________________________
void TGL5D::DrawCloud()const
{
   //Draw full cloud of points.
   if (!fNP || !fV1 || !fV2 || !fV3 || !fV4 || !fV5) {
      //Error("TGL5D::DrawCloud", "Set data sources correctly.");
      return;
   }

   const TGLDisableGuard light(GL_LIGHTING);
   const TGLDisableGuard depth(GL_DEPTH_TEST);
   
   glColor3d(0.4, 0., 1.);
   glPointSize(3.f);
   
   glBegin(GL_POINTS);
   
   const Double_t xs = fCoord->GetXScale();
   const Double_t ys = fCoord->GetYScale();
   const Double_t zs = fCoord->GetZScale();
   
   for (Int_t i = 0; i < fNP; ++i)
      glVertex3d(fV1[i] * xs, fV2[i] * ys, fV3[i] * zs);
   
   glEnd();
   
   glPointSize(1.f);
}

//______________________________________________________________________________
void TGL5D::DrawSubCloud(Double_t v4, Double_t range, Color_t ci)const
{
   //Draw cloud for selected iso-surface.
   if (!fNP || !fV1 || !fV2 || !fV3 || !fV4 || !fV5) {
      Error("TGL5D::DrawSubCloud", "Set data sources correctly.");
      return;
   }
      
   const TGLDisableGuard light(GL_LIGHTING);
   
   Float_t rgb[3] = {};
   Rgl::Pad::ExtractRGB(ci, rgb);
   
   glColor3fv(rgb);
   glPointSize(3.f);
   
   glBegin(GL_POINTS);
   
   const Double_t xs = fCoord->GetXScale();
   const Double_t ys = fCoord->GetYScale();
   const Double_t zs = fCoord->GetZScale();
   
   for (Int_t i = 0; i < fNP; ++i)
      if (TMath::Abs(fV4[i] - v4) < range)
         glVertex3d(fV1[i] * xs, fV2[i] * ys, fV3[i] * zs);
   
   glEnd();
   
   glPointSize(1.f);
}

namespace {
Bool_t InRange(const std::vector<Double_t> &preds, const UInt_t *tri, Double_t min, Double_t max);
}

//______________________________________________________________________________
void TGL5D::DrawMesh(ConstSurfIter_t surf)const
{
   //Draw one iso-surface.
   const Mesh_t &m = surf->fMesh;
   /*
   glEnable(GL_POLYGON_OFFSET_FILL);
   glPolygonOffset(1.f, 1.f);
   */
   if (!fBoxCut.IsActive()) {
      if (!fSelectionPass)
         ::DrawMesh(GL_FLOAT, m.fVerts, m.fNorms, m.fTris);
      else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         ::DrawMesh(GL_FLOAT, m.fVerts, m.fTris);
      }
   } else {
      if (!fSelectionPass) {
         ::DrawMesh(&glNormal3fv, &glVertex3fv, m.fVerts, m.fNorms, m.fTris, fBoxCut);
      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         ::DrawMesh(&glVertex3fv, m.fVerts, m.fTris, fBoxCut);
      }
   }
   /*
   glDisable(GL_POLYGON_OFFSET_FILL);
   
   if (!fSelectionPass) {
      const TGLEnableGuard  blend(GL_BLEND);
      glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
      SetSurfaceColor(kCyan);
      
      glBegin(GL_TRIANGLES);
      
      for (size_type i = 0, e = surf->fMesh.fTris.size() / 3; i < e; ++i) {
         const UInt_t *tri = &surf->fMesh.fTris[i * 3];
         if (InRange(surf->fPreds, tri, fV5SliderMin, fV5SliderMax)) {
            glNormal3fv(&surf->fMesh.fNorms[tri[0] * 3]);
            glVertex3fv(&surf->fMesh.fVerts[tri[0] * 3]);
            glNormal3fv(&surf->fMesh.fNorms[tri[1] * 3]);
            glVertex3fv(&surf->fMesh.fVerts[tri[1] * 3]);
            glNormal3fv(&surf->fMesh.fNorms[tri[2] * 3]);
            glVertex3fv(&surf->fMesh.fVerts[tri[2] * 3]);
         }
      }
      
      glEnd();
   }*/
}

namespace {
//______________________________________________________________________________
void FindRange(Int_t size, const Double_t *src, Rgl::Range_t &range)
{
   range.first = src[0];
   range.second = src[0];
   
   for (Int_t i = 1; i < size; ++i) {
      range.first  = std::min(range.first,  src[i]);
      range.second = std::max(range.second, src[i]);
   }
}

//______________________________________________________________________________
Bool_t InRange(const std::vector<Double_t> &preds, const UInt_t *tri, Double_t min, Double_t max)
{
   for (UInt_t i = 0; i < 3; ++i)
      if (preds[tri[i]] > max || preds[tri[i]] < min)
         return kFALSE;
   
   return kTRUE;
}
}
