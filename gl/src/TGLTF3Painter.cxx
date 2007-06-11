#include "TVirtualGL.h"
#include "KeySymbols.h"
#include "Buttons.h"
#include "TROOT.h"
#include "TColor.h"
#include "TMath.h"
#include "TH1.h"
#include "TF3.h"

#include "TGLOrthoCamera.h"
#include "TGLTF3Painter.h"
#include "TGLIncludes.h"

ClassImp(TGLTF3Painter)

//______________________________________________________________________________
TGLTF3Painter::TGLTF3Painter(TF3 *fun, TH1 *hist, TGLOrthoCamera *camera,
                             TGLPlotCoordinates *coord, Int_t ctx)
                  : TGLPlotPainter(hist, camera, coord, ctx, kTRUE, kTRUE, kTRUE),
                    fStyle(kDefault),
                    fF3(fun),
                    fXOZSlice("XOZ", (TH3 *)hist, fun, coord, &fBackBox, TGLTH3Slice::kXOZ),
                    fYOZSlice("YOZ", (TH3 *)hist, fun, coord, &fBackBox, TGLTH3Slice::kYOZ),
                    fXOYSlice("XOY", (TH3 *)hist, fun, coord, &fBackBox, TGLTH3Slice::kXOY)
{
   // Constructor.
}

//______________________________________________________________________________
char *TGLTF3Painter::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   //Coords for point on surface under cursor.
   return "fun3";
}

namespace {
   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY,
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const TF3 *fun, std::vector<TGLTF3Painter::TriFace_t> &mesh,
                     Rgl::Range_t &minMax);
}

//______________________________________________________________________________
Bool_t TGLTF3Painter::InitGeometry()
{
   //Create mesh.
   fCoord->SetCoordType(kGLCartesian);

   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   fCamera->SetViewVolume(fBackBox.Get3DBox());

   //Build mesh for TF3 surface
   fMesh.clear();

   const Int_t nX = fHist->GetNbinsX();
   const Int_t nY = fHist->GetNbinsY();
   const Int_t nZ = fHist->GetNbinsZ();

   const Double_t xMin = fXAxis->GetBinLowEdge(fXAxis->GetFirst());
   const Double_t xStep = (fXAxis->GetBinUpEdge(fXAxis->GetLast()) - xMin) / nX;
   const Double_t yMin = fYAxis->GetBinLowEdge(fYAxis->GetFirst());
   const Double_t yStep = (fYAxis->GetBinUpEdge(fYAxis->GetLast()) - yMin) / nY;
   const Double_t zMin = fZAxis->GetBinLowEdge(fZAxis->GetFirst());
   const Double_t zStep = (fZAxis->GetBinUpEdge(fZAxis->GetLast()) - zMin) / nZ;

   Rgl::Range_t minMax;
   minMax.first  = fF3->Eval(xMin, yMin, zMin);
   minMax.second = minMax.first;

   for (Int_t i = 0; i < nX; ++i) {
      for (Int_t j= 0; j < nY; ++j) {
         for (Int_t k = 0; k < nZ; ++k) {
            MarchingCube(xMin + i * xStep, yMin + j * yStep, zMin + k * zStep,
                         xStep, yStep, zStep, fCoord->GetXScale(), fCoord->GetYScale(),
                         fCoord->GetZScale(), fF3, fMesh, minMax);
         }
      }
   }

   //Not sure about this part :(
   minMax.second = 0.001 * minMax.first;

   fXOZSlice.SetMinMax(minMax);
   fYOZSlice.SetMinMax(minMax);
   fXOYSlice.SetMinMax(minMax);


   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      const TGLVertex3 &vertex = fBackBox.Get3DBox()[0];
      fXOZSectionPos = vertex.Y();
      fYOZSectionPos = vertex.X();
      fXOYSectionPos = vertex.Z();
      fCoord->ResetModified();
   }


   return kTRUE;
}

//______________________________________________________________________________
void TGLTF3Painter::StartPan(Int_t px, Int_t py)
{
   //User clicks right mouse button (in a pad).
   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}

//______________________________________________________________________________
void TGLTF3Painter::Pan(Int_t px, Int_t py)
{
   //User's moving mouse cursor, with middle mouse button pressed (for pad).
   //Calculate 3d shift related to 2d mouse movement.
   if (!MakeGLContextCurrent())
      return;

   if (fSelectedPart >= fSelectionBase)//Pan camera.
      fCamera->Pan(px, py);
   else if (fSelectedPart > 0) {
      //Convert py into bottom-top orientation.
      //Possibly, move box here
      py = fCamera->GetHeight() - py;
      if (!fHighColor) {
         if (fBoxCut.IsActive() && (fSelectedPart >= kXAxis && fSelectedPart <= kZAxis))
            fBoxCut.MoveBox(px, py, fSelectedPart);
         else
            MoveSection(px, py);
      } else {
         MoveSection(px, py);
      }
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLTF3Painter::AddOption(const TString &/*option*/)
{
   //No options for tf3
}

//______________________________________________________________________________
void TGLTF3Painter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{
   //Change color sheme.
   if (event == kKeyPress) {
      if (py == kKey_s || py == kKey_S) {
         fStyle < kMaple2 ? fStyle = ETF3Style(fStyle + 1) : fStyle = kDefault;
         //gGLManager->PaintSingleObject(this);
      } else if (py == kKey_c || py == kKey_C) {
         if (fHighColor)
            Info("ProcessEvent", "Cut box does not work in high color, please, switch to true color");
         else {
            fBoxCut.TurnOnOff();
            fUpdateSelection = kTRUE;
         }
      }
   } else if (event == kButton1Double && (fBoxCut.IsActive() || HasSections())) {
      if (fBoxCut.IsActive())
         fBoxCut.TurnOnOff();
      const TGLVertex3 *frame = fBackBox.Get3DBox();
      fXOZSectionPos = frame[0].Y();
      fYOZSectionPos = frame[0].X();
      fXOYSectionPos = frame[0].Z();

      gGLManager->PaintSingleObject(this);
   }
}

//______________________________________________________________________________
void TGLTF3Painter::InitGL()const
{
   //Initialize OpenGL state variables.
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

namespace {
   void GetColor(Double_t *color, const TGLVector3 &normal);
}

//______________________________________________________________________________
void TGLTF3Painter::DrawPlot()const
{
   //Draw mesh.
   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);
   DrawSections();

   if (!fSelectionPass && HasSections() && fStyle < kMaple2) {
      //Surface is semi-transparent during dynamic profiling.
      //Having several complex nested surfaces, it's not easy
      //(possible?) to implement correct and _efficient_ transparency
      //drawing. So, artefacts are possbile.
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glDepthMask(GL_FALSE);
   }

   //Draw TF3 surface
   if (!fSelectionPass)
      fStyle > kDefault ? glDisable(GL_LIGHTING) : SetSurfaceColor();//[0

   if (fStyle == kMaple1) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[1
      glPolygonOffset(1.f, 1.f);
   } else if (fStyle == kMaple2)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[2

   if (!fBoxCut.IsActive()) {
      glBegin(GL_TRIANGLES);

      Double_t color[] = {0., 0., 0., 0.15};

      if (!fSelectionPass) {
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glNormal3dv(fMesh[i].fNormals[0].CArr());
            GetColor(color, fMesh[i].fNormals[0]);
            glColor4dv(color);
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glNormal3dv(fMesh[i].fNormals[1].CArr());
            GetColor(color, fMesh[i].fNormals[1]);
            glColor4dv(color);
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glNormal3dv(fMesh[i].fNormals[2].CArr());
            GetColor(color, fMesh[i].fNormals[2]);
            glColor4dv(color);
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }
      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }
      }

      glEnd();

      if (fStyle == kMaple1 && !fSelectionPass) {
         glDisable(GL_POLYGON_OFFSET_FILL);//1]
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[3
         glColor4d(0., 0., 0., 0.25);

         glBegin(GL_TRIANGLES);

         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            glVertex3dv(fMesh[i].fXYZ[0].CArr());
            glVertex3dv(fMesh[i].fXYZ[1].CArr());
            glVertex3dv(fMesh[i].fXYZ[2].CArr());
         }

         glEnd();
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//3]
      } else if (fStyle == kMaple2)
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//2]

      if (fStyle > kDefault && !fSelectionPass)
         glEnable(GL_LIGHTING); //0]
   } else {
      glBegin(GL_TRIANGLES);

      //TGLVector3 color;
      Double_t color[] = {0., 0., 0., 0.15};

      if (!fSelectionPass) {
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());

            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            glNormal3dv(tri.fNormals[0].CArr());
            GetColor(color, tri.fNormals[0]);
            glColor4dv(color);
            glVertex3dv(tri.fXYZ[0].CArr());
            glNormal3dv(tri.fNormals[1].CArr());
            GetColor(color, tri.fNormals[1]);
            glColor4dv(color);
            glVertex3dv(tri.fXYZ[1].CArr());
            glNormal3dv(tri.fNormals[2].CArr());
            GetColor(color, tri.fNormals[2]);
            glColor4dv(color);
            glVertex3dv(tri.fXYZ[2].CArr());
         }
      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());

            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;
            glVertex3dv(tri.fXYZ[0].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }
      }

      glEnd();
      if (fStyle == kMaple1 && !fSelectionPass) {
         glDisable(GL_POLYGON_OFFSET_FILL);//1]
         glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);//[3
         glColor4d(0., 0., 0., 0.25);

         glBegin(GL_TRIANGLES);

         for (UInt_t i = 0, e = fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());

            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;
            glVertex3dv(tri.fXYZ[0].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }

         glEnd();
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//3]
      } else if (fStyle == kMaple2)
         glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);//2]

      if (fStyle > kDefault && !fSelectionPass)
         glEnable(GL_LIGHTING); //0]
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);
   }

   if (!fSelectionPass && HasSections() && fStyle < kMaple2) {
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
   }


}

//______________________________________________________________________________
void TGLTF3Painter::SetSurfaceColor()const
{
   //Set color for surface.
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.15f};

   if (fF3->GetFillColor() != kWhite)
      if (const TColor *c = gROOT->GetColor(fF3->GetFillColor()))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);

   glMaterialfv(GL_BACK, GL_DIFFUSE, diffColor);
   diffColor[0] /= 2, diffColor[1] /= 2, diffColor[2] /= 2;
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}

//______________________________________________________________________________
Bool_t TGLTF3Painter::HasSections()const
{
   //Any section exists.
   return fXOZSectionPos > fBackBox.Get3DBox()[0].Y() || fYOZSectionPos > fBackBox.Get3DBox()[0].X() ||
          fXOYSectionPos > fBackBox.Get3DBox()[0].Z();
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionXOZ()const
{
   // Draw XOZ parallel section.
   if (fSelectionPass)
      return;
   fXOZSlice.DrawSlice(fXOZSectionPos / fCoord->GetYScale());
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionYOZ()const
{
   // Draw YOZ parallel section.
   if (fSelectionPass)
      return;
   fYOZSlice.DrawSlice(fYOZSectionPos / fCoord->GetXScale());
}

//______________________________________________________________________________
void TGLTF3Painter::DrawSectionXOY()const
{
   // Draw XOY parallel section.
   if (fSelectionPass)
      return;
   fXOYSlice.DrawSlice(fXOYSectionPos / fCoord->GetZScale());
}


ClassImp(TGLIsoPainter)

//______________________________________________________________________________
TGLIsoPainter::TGLIsoPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord, Int_t ctx)
                  : TGLPlotPainter(hist, camera, coord, ctx, kTRUE, kTRUE, kTRUE),
                    fXOZSlice("XOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOZ),
                    fYOZSlice("YOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kYOZ),
                    fXOYSlice("XOY", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOY),
                    fInit(kFALSE)
{
   //Constructor.
   if (hist->GetDimension() < 3)
      Error("TGLIsoPainter::TGLIsoPainter", "Wrong type of histogramm, must have 3 dimensions");
}

//______________________________________________________________________________
char *TGLIsoPainter::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   //Return info for plot part under cursor.
   return "iso";
}

namespace {

   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY,
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const Double_t *funValues, std::vector<TGLIsoPainter::TriFace_t> &mesh,
                     Double_t isoValue);

   inline Double_t Abs(Double_t val)
   {
      if(val < 0.) val *= -1.;
      return val;
   }

   inline Bool_t Eq(const TGLVertex3 &v1, const TGLVertex3 &v2)
   {
      return Abs(v1.X() - v2.X()) < 0.0000001 &&
             Abs(v1.Y() - v2.Y()) < 0.0000001 &&
             Abs(v1.Z() - v2.Z()) < 0.0000001;
   }

}

//______________________________________________________________________________
Bool_t TGLIsoPainter::InitGeometry()
{
   //Initializes meshes for 3d iso contours.
   if (fHist->GetDimension() < 3) {
      Error("TGLIsoPainter::TGLIsoPainter", "Wrong type of histogramm, must have 3 dimensions");
      return kFALSE;
   }

   //Create mesh.
   //Now, I check this to avoid
   //expensive recalculations.
   if (fInit)
      return kTRUE;

   //Only in cartesian.
   fCoord->SetCoordType(kGLCartesian);
   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   fCamera->SetViewVolume(fBackBox.Get3DBox());

   //Move old meshed into the cache.
   if (!fIsos.empty())
      fCache.splice(fCache.begin(), fIsos);
   //Number of contours == number of iso surfaces.
   UInt_t nContours = fHist->GetContour();

   if (nContours > 1) {
      fColorLevels.resize(nContours);
      FindMinMax();

      if (fHist->TestBit(TH1::kUserContour)) {
         //There are user defined contours (iso-levels).
         for (UInt_t i = 0; i < nContours; ++i)
            fColorLevels[i] = fHist->GetContourLevelPad(i);
      } else {
         //Equidistant iso-surfaces.
         const Double_t isoStep = (fMinMax.second - fMinMax.first) / nContours;
         for (UInt_t i = 0; i < nContours; ++i)
            fColorLevels[i] = fMinMax.first + i * isoStep;
      }

      fPalette.GeneratePalette(nContours, fMinMax, kFALSE);
   } else {
      //Only one iso (ROOT's standard).
      fColorLevels.resize(nContours = 1);
      fColorLevels[0] = fHist->GetSumOfWeights() / (fHist->GetNbinsX() * fHist->GetNbinsY() * fHist->GetNbinsZ());
   }

   MeshIter_t firstMesh = fCache.begin();
   //Initialize meshes, trying to reuse mesh from
   //mesh cache.
   for (UInt_t i = 0; i < nContours; ++i) {
      if (firstMesh != fCache.end()) {
         //There is a mesh in a chache.
         SetMesh(*firstMesh, fColorLevels[i]);
         MeshIter_t next = firstMesh;
         ++next;
         fIsos.splice(fIsos.begin(), fCache, firstMesh);
         firstMesh = next;
      } else {
         //No meshes in a cache.
         //Create new one and _swap_ data (look at Mesh_t::Swap in a header)
         //between empty mesh in a list and this mesh
         //to avoid real copying.
         Mesh_t newMesh;
         SetMesh(newMesh, fColorLevels[i]);
         fIsos.push_back(fDummyMesh);
         fIsos.back().Swap(newMesh);
      }
   }


   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      fXOZSectionPos = fBackBox.Get3DBox()[0].Y();
      fYOZSectionPos = fBackBox.Get3DBox()[0].X();
      fXOYSectionPos = fBackBox.Get3DBox()[0].Z();
      fCoord->ResetModified();
   }

   //Avoid rebuilding the mesh.
   fInit = kTRUE;

   return kTRUE;

}

//______________________________________________________________________________
void TGLIsoPainter::StartPan(Int_t px, Int_t py)
{
   //User clicks right mouse button (in a pad).
   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}

//______________________________________________________________________________
void TGLIsoPainter::Pan(Int_t px, Int_t py)
{
   //User's moving mouse cursor, with middle mouse button pressed (for pad).
   //Calculate 3d shift related to 2d mouse movement.
   // User's moving mouse cursor, with middle mouse button pressed (for pad).
   // Calculate 3d shift related to 2d mouse movement.

   if (!MakeGLContextCurrent())
      return;

   if (fSelectedPart >= fSelectionBase)//Pan camera.
      fCamera->Pan(px, py);
   else if (fSelectedPart > 0) {
      //Convert py into bottom-top orientation.
      //Possibly, move box here
      py = fCamera->GetHeight() - py;
      if (!fHighColor) {
         if (fBoxCut.IsActive() && (fSelectedPart >= kXAxis && fSelectedPart <= kZAxis))
            fBoxCut.MoveBox(px, py, fSelectedPart);
         else
            MoveSection(px, py);
      } else {
         MoveSection(px, py);
      }
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLIsoPainter::AddOption(const TString &/*option*/)
{
   //No additional options for TGLIsoPainter.
}

//______________________________________________________________________________
void TGLIsoPainter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
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
      }
   } else if (event == kButton1Double && (fBoxCut.IsActive() || HasSections())) {
      if (fBoxCut.IsActive())
         fBoxCut.TurnOnOff();
      const TGLVertex3 *frame = fBackBox.Get3DBox();
      fXOZSectionPos = frame[0].Y();
      fYOZSectionPos = frame[0].X();
      fXOYSectionPos = frame[0].Z();

      gGLManager->PaintSingleObject(this);
   }
}

//______________________________________________________________________________
void TGLIsoPainter::InitGL()const
{
   //Initialize OpenGL state variables.
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_DEPTH_TEST);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGLIsoPainter::DrawPlot()const
{
   //Draw mesh.
   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);
   DrawSections();

   if (fIsos.size() != fColorLevels.size()) {
      Error("TGLIsoPainter::DrawPlot", "Non-equal number of levels and isos");
      return;
   }

   if (!fSelectionPass && HasSections()) {
      //Surface is semi-transparent during dynamic profiling.
      //Having several complex nested surfaces, it's not easy
      //(possible?) to implement correct and _efficient_ transparency
      //drawing. So, artefacts are possbile.
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glDepthMask(GL_FALSE);
   }

   UInt_t colorInd = 0;
   ConstMeshIter_t iso = fIsos.begin();

   for (; iso != fIsos.end(); ++iso, ++colorInd)
      DrawMesh(*iso, colorInd);

   if (!fSelectionPass && HasSections()) {
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
   }

   if (fBoxCut.IsActive())
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);
}

//______________________________________________________________________________
void TGLIsoPainter::DrawSectionXOZ()const
{
   // Draw XOZ parallel section.
   if (fSelectionPass)
      return;
   fXOZSlice.DrawSlice(fXOZSectionPos / fCoord->GetYScale());
}

//______________________________________________________________________________
void TGLIsoPainter::DrawSectionYOZ()const
{
   // Draw YOZ parallel section.
   if (fSelectionPass)
      return;
   fYOZSlice.DrawSlice(fYOZSectionPos / fCoord->GetXScale());
}

//______________________________________________________________________________
void TGLIsoPainter::DrawSectionXOY()const
{
   // Draw XOY parallel section.
   if (fSelectionPass)
      return;
   fXOYSlice.DrawSlice(fXOYSectionPos / fCoord->GetZScale());
}

//______________________________________________________________________________
Bool_t TGLIsoPainter::HasSections()const
{
   //Any section exists.
   return fXOZSectionPos > fBackBox.Get3DBox()[0].Y() || fYOZSectionPos > fBackBox.Get3DBox()[0].X() ||
          fXOYSectionPos > fBackBox.Get3DBox()[0].Z();
}

//______________________________________________________________________________
void TGLIsoPainter::SetSurfaceColor(Int_t ind)const
{
   //Set color for surface.
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.25f};

   if (fColorLevels.size() == 1) {
      if (fHist->GetFillColor() != kWhite)
         if (const TColor *c = gROOT->GetColor(fHist->GetFillColor()))
            c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   } else {
      const UChar_t *color = fPalette.GetColour(ind);
      diffColor[0] = color[0] / 255.;
      diffColor[1] = color[1] / 255.;
      diffColor[2] = color[2] / 255.;
   }

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   diffColor[0] /= 3.5, diffColor[1] /= 3.5, diffColor[2] /= 3.5;
   glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, diffColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30.f);
}

//______________________________________________________________________________
void TGLIsoPainter::SetMesh(Mesh_t &m, Double_t isoValue)
{
   //Set mesh for iso surface at level isoValue.
   //Large and nightmarish "unrolled" code - I'm doing simple optimisation:
   //marching cubes calculates a set of triangles (possible empty)
   //for each of cubes in a lattice. After that, I need to calculate
   //per-vertex smoothed normals - calculating the summ of neighbouring
   //per-triangle normals and normalizing
   //(so, I need to find common vertices for triangles),
   //this can be done only after
   //each of 26 neighbouring cubes was processed. I remember
   //"mesh range" for each cube, not to check _EVERY_ triangles.
   const Int_t nX = fHist->GetNbinsX();
   const Int_t nY = fHist->GetNbinsY();
   const Int_t nZ = fHist->GetNbinsZ();

   const Double_t xMin      = fXAxis->GetBinCenter(fXAxis->GetFirst());
   const Double_t xStep     = (fXAxis->GetBinCenter(fXAxis->GetLast()) - xMin) / (nX - 1);
   const Double_t yMin      = fYAxis->GetBinCenter(fYAxis->GetFirst());
   const Double_t yStep     = (fYAxis->GetBinCenter(fYAxis->GetLast()) - yMin) / (nY - 1);
   const Double_t zMin      = fZAxis->GetBinCenter(fZAxis->GetFirst());
   const Double_t zStep     = (fZAxis->GetBinCenter(fZAxis->GetLast()) - zMin) / (nZ - 1);
   const Int_t    sliceSize = (nY + 2) * (nZ + 2);
   std::vector<Range_t> &boxRanges = m.fBoxRanges;
   std::vector<TriFace_t> &mesh = m.fMesh;

   boxRanges.assign((nX + 2) * (nY + 2) * (nZ + 2), Range_t());

   //First, calculate full mesh and define "box ranges" - which
   //part of the full mesh is in current box.
   //Find flat normals.
   for (Int_t i = 0, ir = fXAxis->GetFirst(), ei = fXAxis->GetLast(); ir < ei; ++i, ++ir) {
      for (Int_t j = 0, jr = fYAxis->GetFirst(), ej = fYAxis->GetLast(); jr < ej; ++j, ++jr) {
         for (Int_t k = 0, kr = fZAxis->GetFirst(), ek = fZAxis->GetLast(); kr < ek; ++k, ++kr) {
            const Double_t cube[] = {fHist->GetBinContent(ir, jr, kr),             fHist->GetBinContent(ir + 1, jr, kr),
                                     fHist->GetBinContent(ir + 1, jr + 1, kr),     fHist->GetBinContent(ir, jr + 1, kr),
                                     fHist->GetBinContent(ir, jr, kr + 1),         fHist->GetBinContent(ir + 1, jr, kr + 1),
                                     fHist->GetBinContent(ir + 1, jr + 1, kr + 1), fHist->GetBinContent(ir, jr + 1, kr + 1)};
            Int_t start  = Int_t(mesh.size());
            MarchingCube(xMin + i * xStep, yMin + j * yStep, zMin + k * zStep, xStep, yStep, zStep,
                         fCoord->GetXScale(), fCoord->GetYScale(), fCoord->GetZScale(), cube, mesh,
                         isoValue);
            Int_t finish = Int_t(mesh.size());
            if (start != finish)
               boxRanges[(ir + 1) * sliceSize + (jr + 1) * (nZ + 2) + kr + 1] = Range_t(start, finish);
         }
      }
   }

   for (Int_t i = 1; i <= nX; ++i) {
      for (Int_t j = 1; j <= nY; ++j) {
         for (Int_t k = 1; k <= nZ; ++k) {
            Range_t &box = boxRanges[i * sliceSize + j * (nZ + 2) + k];
            if (box.fFirst != -1) {
               for (Int_t tri = box.fFirst; tri < box.fLast; ++tri) {
                  TriFace_t &face = mesh[tri];
                  //First, check triangles from the same box.
                  for (Int_t k1 = 0; k1 < 3; ++k1) face.fPerVertexNormals[k1] = face.fNormal;
                  const TGLVertex3 &v0 = face.fXYZ[0];
                  const TGLVertex3 &v1 = face.fXYZ[1];
                  const TGLVertex3 &v2 = face.fXYZ[2];

                  for (Int_t tri1 = box.fFirst; tri1 < box.fLast; ++tri1) {
                     if (tri != tri1) {
                        const TriFace_t &testFace = mesh[tri1];
                        if (Eq(v0, testFace.fXYZ[0]))
                           face.fPerVertexNormals[0] += testFace.fNormal;
                        if (Eq(v0, testFace.fXYZ[1]))
                           face.fPerVertexNormals[0] += testFace.fNormal;
                        if (Eq(v0, testFace.fXYZ[2]))
                           face.fPerVertexNormals[0] += testFace.fNormal;
                        if (Eq(v1, testFace.fXYZ[0]))
                           face.fPerVertexNormals[1] += testFace.fNormal;
                        if (Eq(v1, testFace.fXYZ[1]))
                           face.fPerVertexNormals[1] += testFace.fNormal;
                        if (Eq(v1, testFace.fXYZ[2]))
                           face.fPerVertexNormals[1] += testFace.fNormal;
                        if (Eq(v2, testFace.fXYZ[0]))
                           face.fPerVertexNormals[2] += testFace.fNormal;
                        if (Eq(v2, testFace.fXYZ[1]))
                           face.fPerVertexNormals[2] += testFace.fNormal;
                        if (Eq(v2, testFace.fXYZ[2]))
                           face.fPerVertexNormals[2] += testFace.fNormal;
                     }
                  }

                  const Int_t nZ2 = nZ + 2;

                  Range_t &box1  = boxRanges[(i - 1) * sliceSize + (j - 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box1);
                  Range_t &box2  = boxRanges[(i) * sliceSize + (j - 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box2);
                  Range_t &box3  = boxRanges[(i + 1) * sliceSize + (j - 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box3);
                  Range_t &box4  = boxRanges[(i + 1) * sliceSize + (j) * nZ2 + k - 1];
                  CheckBox(mesh, face, box4);
                  Range_t &box5  = boxRanges[(i + 1) * sliceSize + (j + 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box5);
                  Range_t &box6  = boxRanges[(i) * sliceSize + (j + 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box6);
                  Range_t &box7  = boxRanges[(i - 1) * sliceSize + (j + 1) * nZ2 + k - 1];
                  CheckBox(mesh, face, box7);
                  Range_t &box8  = boxRanges[(i - 1) * sliceSize + (j) * nZ2 + k - 1];
                  CheckBox(mesh, face, box8);
                  Range_t &box9  = boxRanges[(i) * sliceSize + (j) * nZ2 + k - 1];
                  CheckBox(mesh, face, box9);

                  Range_t &box10  = boxRanges[(i - 1) * sliceSize + (j - 1) * nZ2 + k];
                  CheckBox(mesh, face, box10);
                  Range_t &box11  = boxRanges[(i) * sliceSize + (j - 1) * nZ2 + k];
                  CheckBox(mesh, face, box11);
                  Range_t &box12  = boxRanges[(i + 1) * sliceSize + (j - 1) * nZ2 + k];
                  CheckBox(mesh, face, box12);
                  Range_t &box13  = boxRanges[(i + 1) * sliceSize + (j) * nZ2 + k];
                  CheckBox(mesh, face, box13);
                  Range_t &box14  = boxRanges[(i + 1) * sliceSize + (j + 1) * nZ2 + k];
                  CheckBox(mesh, face, box14);
                  Range_t &box15  = boxRanges[(i) * sliceSize + (j + 1) * nZ2 + k];
                  CheckBox(mesh, face, box15);
                  Range_t &box16  = boxRanges[(i - 1) * sliceSize + (j + 1) * nZ2 + k];
                  CheckBox(mesh, face, box16);
                  Range_t &box17  = boxRanges[(i - 1) * sliceSize + (j) * nZ2 + k];
                  CheckBox(mesh, face, box17);

                  Range_t &box18  = boxRanges[(i - 1) * sliceSize + (j - 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box18);
                  Range_t &box19  = boxRanges[(i) * sliceSize + (j - 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box19);
                  Range_t &box20  = boxRanges[(i + 1) * sliceSize + (j - 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box20);
                  Range_t &box21  = boxRanges[(i + 1) * sliceSize + (j) * nZ2 + k + 1];
                  CheckBox(mesh, face, box21);
                  Range_t &box22  = boxRanges[(i + 1) * sliceSize + (j + 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box22);
                  Range_t &box23  = boxRanges[(i) * sliceSize + (j + 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box23);
                  Range_t &box24  = boxRanges[(i - 1) * sliceSize + (j + 1) * nZ2 + k + 1];
                  CheckBox(mesh, face, box24);
                  Range_t &box25  = boxRanges[(i - 1) * sliceSize + (j) * nZ2 + k + 1];
                  CheckBox(mesh, face, box25);
                  Range_t &box26  = boxRanges[(i) * sliceSize + (j) * nZ2 + k + 1];
                  CheckBox(mesh, face, box26);
               }
            }
         }
      }
   }

   for (UInt_t i = 0, ei = mesh.size(); i < ei; ++i) {
      TriFace_t &face = mesh[i];
      if(face.fPerVertexNormals[0].X() || face.fPerVertexNormals[0].Y() || face.fPerVertexNormals[0].Z())
         face.fPerVertexNormals[0].Normalise();
      if(face.fPerVertexNormals[1].X() || face.fPerVertexNormals[1].Y() || face.fPerVertexNormals[1].Z())
         face.fPerVertexNormals[1].Normalise();
      if(face.fPerVertexNormals[2].X() || face.fPerVertexNormals[2].Y() || face.fPerVertexNormals[2].Z())
         face.fPerVertexNormals[2].Normalise();
   }

}

//______________________________________________________________________________
void TGLIsoPainter::DrawMesh(const Mesh_t &mesh, Int_t level)const
{
   //Draw TF3 surface
   if (!fSelectionPass)
      SetSurfaceColor(level);

   if (!fBoxCut.IsActive()) {
      glBegin(GL_TRIANGLES);


      if (!fSelectionPass) {
         for (UInt_t i = 0, e = mesh.fMesh.size(); i < e; ++i) {
            glNormal3dv(mesh.fMesh[i].fPerVertexNormals[0].CArr());
            glVertex3dv(mesh.fMesh[i].fXYZ[0].CArr());
            glNormal3dv(mesh.fMesh[i].fPerVertexNormals[1].CArr());
            glVertex3dv(mesh.fMesh[i].fXYZ[1].CArr());
            glNormal3dv(mesh.fMesh[i].fPerVertexNormals[2].CArr());
            glVertex3dv(mesh.fMesh[i].fXYZ[2].CArr());
         }

      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = mesh.fMesh.size(); i < e; ++i) {
            glVertex3dv(mesh.fMesh[i].fXYZ[0].CArr());
            glVertex3dv(mesh.fMesh[i].fXYZ[1].CArr());
            glVertex3dv(mesh.fMesh[i].fXYZ[2].CArr());
         }
      }

      glEnd();

   } else {
      glBegin(GL_TRIANGLES);

      if (!fSelectionPass) {
         for (UInt_t i = 0, e = mesh.fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = mesh.fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());

            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            glNormal3dv(tri.fPerVertexNormals[0].CArr());
            glVertex3dv(tri.fXYZ[0].CArr());
            glNormal3dv(tri.fPerVertexNormals[1].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glNormal3dv(tri.fPerVertexNormals[2].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }

      } else {
         Rgl::ObjectIDToColor(fSelectionBase, fHighColor);
         for (UInt_t i = 0, e = mesh.fMesh.size(); i < e; ++i) {
            const TriFace_t &tri = mesh.fMesh[i];
            const Double_t xMin = TMath::Min(TMath::Min(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t xMax = TMath::Max(TMath::Max(tri.fXYZ[0].X(), tri.fXYZ[1].X()), tri.fXYZ[2].X());
            const Double_t yMin = TMath::Min(TMath::Min(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t yMax = TMath::Max(TMath::Max(tri.fXYZ[0].Y(), tri.fXYZ[1].Y()), tri.fXYZ[2].Y());
            const Double_t zMin = TMath::Min(TMath::Min(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());
            const Double_t zMax = TMath::Max(TMath::Max(tri.fXYZ[0].Z(), tri.fXYZ[1].Z()), tri.fXYZ[2].Z());

            if (fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            glVertex3dv(tri.fXYZ[0].CArr());
            glVertex3dv(tri.fXYZ[1].CArr());
            glVertex3dv(tri.fXYZ[2].CArr());
         }
      }

      glEnd();
   }
}

//______________________________________________________________________________
void TGLIsoPainter::CheckBox(const std::vector<TriFace_t> &mesh, TriFace_t &face, const Range_t &box)
{
   //For given box and given fase, check if any of box faces has
   //common vertex with face, if yes - att its flat normal.
   if (box.fFirst == -1)
      return;

   const TGLVertex3 &v0 = face.fXYZ[0];
   const TGLVertex3 &v1 = face.fXYZ[1];
   const TGLVertex3 &v2 = face.fXYZ[2];

   for (Int_t tri1 = box.fFirst; tri1 < box.fLast; ++tri1) {
      const TriFace_t &testFace = mesh[tri1];
      if (Eq(v0, testFace.fXYZ[0]))
         face.fPerVertexNormals[0] += testFace.fNormal;
      if (Eq(v0, testFace.fXYZ[1]))
         face.fPerVertexNormals[0] += testFace.fNormal;
      if (Eq(v0, testFace.fXYZ[2]))
         face.fPerVertexNormals[0] += testFace.fNormal;
      if (Eq(v1, testFace.fXYZ[0]))
         face.fPerVertexNormals[1] += testFace.fNormal;
      if (Eq(v1, testFace.fXYZ[1]))
         face.fPerVertexNormals[1] += testFace.fNormal;
      if (Eq(v1, testFace.fXYZ[2]))
         face.fPerVertexNormals[1] += testFace.fNormal;
      if (Eq(v2, testFace.fXYZ[0]))
         face.fPerVertexNormals[2] += testFace.fNormal;
      if (Eq(v2, testFace.fXYZ[1]))
         face.fPerVertexNormals[2] += testFace.fNormal;
      if (Eq(v2, testFace.fXYZ[2]))
         face.fPerVertexNormals[2] += testFace.fNormal;
   }
}

//______________________________________________________________________________
void TGLIsoPainter::FindMinMax()
{
   //Find max/min bin contents for TH3.
   fMinMax.first  = fHist->GetBinContent(fXAxis->GetFirst(), fYAxis->GetFirst(), fZAxis->GetFirst());
   fMinMax.second = fMinMax.first;

   for (Int_t i = fXAxis->GetFirst(), ei = fXAxis->GetLast(); i <= ei; ++i) {
      for (Int_t j = fYAxis->GetFirst(), ej = fYAxis->GetLast(); j <= ej; ++j) {
         for (Int_t k = fZAxis->GetFirst(), ek = fZAxis->GetLast(); k <= ek; ++k) {
            const Double_t binContent = fHist->GetBinContent(i, j, k);
            fMinMax.first  = TMath::Min(binContent, fMinMax.first);
            fMinMax.second = TMath::Max(binContent, fMinMax.second);
         }
      }
   }
}

/*
TF3's based on a small, nice and neat implementation of marching cubes by Cory Bloyd (corysama at yahoo.com)
(many thanks!!!). All possible errors in code are mine - I've modified original code. (tpochep)
*/


namespace {
   //These tables are used so that everything can be done in little loops that you can look at all at once
   // rather than in pages and pages of unrolled code.
   //gA2VertexOffset lists the positions, relative to vertex0, of each of the 8 vertices of a cube
   const Double_t gA2VertexOffset[8][3] =
   {
      {0., 0., 0.}, {1., 0., 0.}, {1., 1., 0.},
      {0., 1., 0.}, {0., 0., 1.}, {1., 0., 1.},
      {1., 1., 1.}, {0., 1., 1.}
   };
   //gA2EdgeConnection lists the index of the endpoint vertices for each of the 12 edges of the cube
   const Int_t gA2EdgeConnection[12][2] =
   {
      {0, 1}, {1, 2}, {2, 3}, {3, 0},
      {4, 5}, {5, 6}, {6, 7}, {7, 4},
      {0,4}, {1,5}, {2,6}, {3,7}
   };
   //gA2EdgeDirection lists the direction vector (vertex1-vertex0) for each edge in the cube
   const Double_t gA2EdgeDirection[12][3] =
   {
      {1., 0., 0.}, {0., 1., 0.}, {-1., 0., 0.},
      {0., -1., 0.}, {1., 0., 0.}, {0., 1., 0.},
      {-1., 0., 0.}, {0., -1., 0.}, {0., 0., 1.},
      {0., 0., 1.}, { 0., 0., 1.}, {0., 0., 1.}
   };

   const Float_t gTargetValue = 0.2f;
   //GetOffset finds the approximate point of intersection of the surface
   // between two points with the values fValue1 and fValue2
   Double_t GetOffset(Double_t val1, Double_t val2, Double_t valDesired)
   {
      Double_t delta = val2 - val1;

      if (!delta)
         return 0.5;

      return (valDesired - val1) / delta;
   }

   //GetColor generates a color from a given normal
   void GetColor(Double_t *rfColor, const TGLVector3 &normal)
   {
      Double_t x = normal.X();
      Double_t y = normal.Y();
      Double_t z = normal.Z();
      rfColor[0] = (x > 0. ? x : 0.) + (y < 0. ? -0.5 * y : 0.) + (z < 0. ? -0.5 * z : 0.);
      rfColor[1] = (y > 0. ? y : 0.) + (z < 0. ? -0.5 * z : 0.) + (x < 0. ? -0.5 * x : 0.);
      rfColor[2] = (z > 0. ? z : 0.) + (x < 0. ? -0.5 * x : 0.) + (y < 0. ? -0.5 * y : 0.);
   }

   void GetNormal(TGLVector3 &normal, Double_t x, Double_t y, Double_t z, const TF3 *fun)
   {
      normal.X() = fun->Eval(x - 0.01, y, z) - fun->Eval(x + 0.01, y, z);
      normal.Y() = fun->Eval(x, y - 0.01, z) - fun->Eval(x, y + 0.01, z);
      normal.Z() = fun->Eval(x, y, z - 0.01) - fun->Eval(x, y, z + 0.01);
      normal.Normalise();
   }

   void GetNormal(TGLIsoPainter::TriFace_t &face)
   {
      TMath::Normal2Plane(face.fXYZ[0].CArr(), face.fXYZ[1].CArr(), face.fXYZ[2].CArr(), face.fNormal.Arr());
   }

   extern Int_t gCubeEdgeFlags[256];
   extern Int_t gTriangleConnectionTable[256][16];

   //MarchingCube performs the Marching Cubes algorithm on a single cube
   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY,
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const TF3 *fun, std::vector<TGLTF3Painter::TriFace_t> &mesh,
                     Rgl::Range_t &minMax)
   {
      Double_t afCubeValue[8] = {0.};
      TGLVector3 asEdgeVertex[12];
      TGLVector3 asEdgeNorm[12];

      //Make a local copy of the values at the cube's corners
      for (Int_t iVertex = 0; iVertex < 8; ++iVertex) {
         afCubeValue[iVertex] = fun->Eval(x + gA2VertexOffset[iVertex][0] * stepX,
                                          y + gA2VertexOffset[iVertex][1] * stepY,
                                          z + gA2VertexOffset[iVertex][2] * stepZ);
         minMax.first  = TMath::Min(minMax.first,  afCubeValue[iVertex]);
         minMax.second = TMath::Max(minMax.second, afCubeValue[iVertex]);
      }

      //Find which vertices are inside of the surface and which are outside
      Int_t iFlagIndex = 0;

      for (Int_t iVertexTest = 0; iVertexTest < 8; ++iVertexTest) {
         if(afCubeValue[iVertexTest] <= gTargetValue)
            iFlagIndex |= 1<<iVertexTest;
      }

      //Find which edges are intersected by the surface
      Int_t iEdgeFlags = gCubeEdgeFlags[iFlagIndex];
      //If the cube is entirely inside or outside of the surface, then there will be no intersections
      if (!iEdgeFlags) return;
      //Find the point of intersection of the surface with each edge
      //Then find the normal to the surface at those points
      for (Int_t iEdge = 0; iEdge < 12; ++iEdge) {
         //if there is an intersection on this edge
         if (iEdgeFlags & (1<<iEdge)) {
            Double_t offset = GetOffset(afCubeValue[ gA2EdgeConnection[iEdge][0] ],
                                       afCubeValue[ gA2EdgeConnection[iEdge][1] ],
                                       gTargetValue);

            asEdgeVertex[iEdge].X() = x + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][0]  +  offset * gA2EdgeDirection[iEdge][0]) * stepX;
            asEdgeVertex[iEdge].Y() = y + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][1]  +  offset * gA2EdgeDirection[iEdge][1]) * stepY;
            asEdgeVertex[iEdge].Z() = z + (gA2VertexOffset[ gA2EdgeConnection[iEdge][0] ][2]  +  offset * gA2EdgeDirection[iEdge][2]) * stepZ;

            GetNormal(asEdgeNorm[iEdge], asEdgeVertex[iEdge].X(), asEdgeVertex[iEdge].Y(), asEdgeVertex[iEdge].Z(), fun);
         }
      }

      //Draw the triangles that were found.  There can be up to five per cube
      for (Int_t iTriangle = 0; iTriangle < 5; iTriangle++) {
         if(gTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
            break;

         TGLTF3Painter::TriFace_t newTri;

         for (Int_t iCorner = 2; iCorner >= 0; --iCorner) {
            Int_t iVertex = gTriangleConnectionTable[iFlagIndex][3*iTriangle+iCorner];

            newTri.fXYZ[iCorner].X() = asEdgeVertex[iVertex].X() * scaleX;
            newTri.fXYZ[iCorner].Y() = asEdgeVertex[iVertex].Y() * scaleY;
            newTri.fXYZ[iCorner].Z() = asEdgeVertex[iVertex].Z() * scaleZ;

            newTri.fNormals[iCorner] = asEdgeNorm[iVertex];
         }

         mesh.push_back(newTri);
      }
   }

   //MarchingCube performs the Marching Cubes algorithm on a single cube
   void MarchingCube(Double_t x, Double_t y, Double_t z, Double_t stepX, Double_t stepY,
                     Double_t stepZ, Double_t scaleX, Double_t scaleY, Double_t scaleZ,
                     const Double_t *cube, std::vector<TGLIsoPainter::TriFace_t> &mesh, Double_t isoValue)
   {
      TGLVector3 asEdgeVertex[12];

      //Find which vertices are inside of the surface and which are outside
      Int_t iFlagIndex = 0;

      for (Int_t iVertexTest = 0; iVertexTest < 8; ++iVertexTest) {
         if(cube[iVertexTest] <= isoValue)
            iFlagIndex |= 1 << iVertexTest;
      }

      //Find which edges are intersected by the surface
      Int_t iEdgeFlags = gCubeEdgeFlags[iFlagIndex];
      //If the cube is entirely inside or outside of the surface, then there will be no intersections
      if (!iEdgeFlags)
         return;

      //Find the point of intersection of the surface with each edge
      //Then find the normal to the surface at those points
      for (Int_t iEdge = 0; iEdge < 12; ++iEdge) {
         //if there is an intersection on this edge
         if (iEdgeFlags & (1<<iEdge)) {
            Double_t offset = GetOffset(cube[gA2EdgeConnection[iEdge][0]], cube[gA2EdgeConnection[iEdge][1]], isoValue);

            asEdgeVertex[iEdge].X() = x + (gA2VertexOffset[gA2EdgeConnection[iEdge][0]][0] + offset * gA2EdgeDirection[iEdge][0]) * stepX;
            asEdgeVertex[iEdge].Y() = y + (gA2VertexOffset[gA2EdgeConnection[iEdge][0]][1] + offset * gA2EdgeDirection[iEdge][1]) * stepY;
            asEdgeVertex[iEdge].Z() = z + (gA2VertexOffset[gA2EdgeConnection[iEdge][0]][2] + offset * gA2EdgeDirection[iEdge][2]) * stepZ;
         }
      }

      //Draw the triangles that were found.  There can be up to five per cube
      for (Int_t iTriangle = 0; iTriangle < 5; iTriangle++) {
         if(gTriangleConnectionTable[iFlagIndex][3 * iTriangle] < 0)
            break;

         TGLIsoPainter::TriFace_t newTri;

         for (Int_t iCorner = 2; iCorner >= 0; --iCorner) {
            Int_t iVertex = gTriangleConnectionTable[iFlagIndex][3 * iTriangle + iCorner];

            newTri.fXYZ[iCorner].X() = asEdgeVertex[iVertex].X() * scaleX;
            newTri.fXYZ[iCorner].Y() = asEdgeVertex[iVertex].Y() * scaleY;
            newTri.fXYZ[iCorner].Z() = asEdgeVertex[iVertex].Z() * scaleZ;
         }

         GetNormal(newTri);

         mesh.push_back(newTri);
      }
   }

   // For any edge, if one vertex is inside of the surface and the other is outside of the surface
   //  then the edge intersects the surface
   // For each of the 8 vertices of the cube can be two possible states : either inside or outside of the surface
   // For any cube the are 2^8=256 possible sets of vertex states
   // This table lists the edges intersected by the surface for all 256 possible vertex states
   // There are 12 edges.  For each entry in the table, if edge #n is intersected, then bit #n is set to 1

   Int_t gCubeEdgeFlags[256]=
   {
      0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c, 0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
      0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c, 0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
      0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c, 0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
      0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac, 0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
      0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c, 0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
      0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc, 0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
      0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c, 0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
      0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc, 0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
      0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc, 0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
      0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c, 0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
      0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc, 0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
      0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c, 0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
      0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac, 0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
      0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c, 0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
      0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c, 0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
      0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c, 0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000
   };

   //  For each of the possible vertex states listed in gCubeEdgeFlags there is a specific triangulation
   //  of the edge intersection points.  gTriangleConnectionTable lists all of them in the form of
   //  0-5 edge triples with the list terminated by the invalid value -1.
   //  For example: gTriangleConnectionTable[3] list the 2 triangles formed when corner[0]
   //  and corner[1] are inside of the surface, but the rest of the cube is not.
   //
   //  I found this table in an example program someone wrote long ago.  It was probably generated by hand

   GLint gTriangleConnectionTable[256][16] =
   {
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
      {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
      {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
      {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
      {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
      {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
      {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
      {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
      {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
      {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
      {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
      {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
      {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
      {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
      {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
      {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
      {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
      {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
      {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
      {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
      {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
      {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
      {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
      {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
      {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
      {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
      {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
      {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
      {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
      {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
      {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
      {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
      {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
      {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
      {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
      {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
      {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
      {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
      {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
      {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
      {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
      {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
      {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
      {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
      {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
      {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
      {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
      {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
      {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
      {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
      {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
      {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
      {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
      {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
      {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
      {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
      {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
      {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
      {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
      {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
      {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
      {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
      {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
      {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
      {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
      {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
      {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
      {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
      {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
      {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
      {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
      {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
      {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
      {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
      {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
      {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
      {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
      {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
      {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
      {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
      {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
      {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
      {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
      {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
      {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
      {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
      {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
      {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
      {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
      {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
      {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
      {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
      {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
      {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
      {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
      {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
      {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
      {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
      {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
      {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
      {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
      {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
      {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
      {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
      {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
      {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
      {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
      {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
      {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
      {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
      {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
      {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
      {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
      {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
      {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
      {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
      {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
      {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
      {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
      {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
      {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
      {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
      {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
      {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
      {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
      {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
      {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
      {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
      {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}
   };

}
