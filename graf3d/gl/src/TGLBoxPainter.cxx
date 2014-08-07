// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  31/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <cctype>

#include "KeySymbols.h"
#include "TVirtualX.h"
#include "Buttons.h"
#include "TString.h"
#include "TROOT.h"
#include "TClass.h"
#include "TColor.h"
#include "TStyle.h"
#include "TH3.h"
#include "TVirtualMutex.h"

#include "TPolyMarker3D.h"
#include "TGLPlotCamera.h"
#include "TGLBoxPainter.h"
#include "TGLIncludes.h"

//______________________________________________________________________________
//
// Paints TH3 histograms by rendering variable-sized bozes matching the
// bin contents.

ClassImp(TGLBoxPainter)

//______________________________________________________________________________
TGLBoxPainter::TGLBoxPainter(TH1 *hist, TGLPlotCamera *cam, TGLPlotCoordinates *coord)
                  : TGLPlotPainter(hist, cam, coord, kTRUE, kTRUE, kTRUE),
                    fXOZSlice("XOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOZ),
                    fYOZSlice("YOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kYOZ),
                    fXOYSlice("XOY", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOY),
                    fType(kBox),
                    fPolymarker(0)
{
   // Normal constructor.
}


//______________________________________________________________________________
TGLBoxPainter::TGLBoxPainter(TH1 *hist, TPolyMarker3D * pm,
                             TGLPlotCamera *cam, TGLPlotCoordinates *coord)
                  : TGLPlotPainter(hist, cam, coord, kFALSE, kFALSE, kFALSE),
                    fXOZSlice("XOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOZ),
                    fYOZSlice("YOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kYOZ),
                    fXOYSlice("XOY", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOY),
                    fType(kBox),
                    fPolymarker(pm)
{
   // Normal constructor.
}

//______________________________________________________________________________
char *TGLBoxPainter::GetPlotInfo(Int_t, Int_t)
{
   //Show box info (i, j, k, binContent).

   fPlotInfo = "";

   if (fSelectedPart) {
      if (fSelectedPart < fSelectionBase) {
         if (fHist->Class())
            fPlotInfo += fHist->Class()->GetName();
         fPlotInfo += "::";
         fPlotInfo += fHist->GetName();
      } else if (!fHighColor){
         const Int_t arr2Dsize = fCoord->GetNYBins() * fCoord->GetNZBins();
         const Int_t binI = (fSelectedPart - fSelectionBase) / arr2Dsize + fCoord->GetFirstXBin();
         const Int_t binJ = (fSelectedPart - fSelectionBase) % arr2Dsize / fCoord->GetNZBins() + fCoord->GetFirstYBin();
         const Int_t binK = (fSelectedPart - fSelectionBase) % arr2Dsize % fCoord->GetNZBins() + fCoord->GetFirstZBin();

         fPlotInfo.Form("(binx = %d; biny = %d; binz = %d; binc = %f)", binI, binJ, binK,
                        fHist->GetBinContent(binI, binJ, binK));
      } else
         fPlotInfo = "Switch to true color mode to get correct info";
   }

   return (Char_t *)fPlotInfo.Data();
}


//______________________________________________________________________________
Bool_t TGLBoxPainter::InitGeometry()
{
  //Set ranges, find min and max bin content.

   fCoord->SetZLog(kFALSE);
   fCoord->SetYLog(kFALSE);
   fCoord->SetXLog(kFALSE);

   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))//kFALSE == drawErrors, kTRUE == zAsBins
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   if(fCamera) fCamera->SetViewVolume(fBackBox.Get3DBox());

   fMinMaxVal.second  = fHist->GetBinContent(fCoord->GetFirstXBin(), fCoord->GetFirstYBin(), fCoord->GetFirstZBin());
   fMinMaxVal.first = fMinMaxVal.second;
   //Bad. You can up-date some bin value and get wrong picture.
   for (Int_t ir = fCoord->GetFirstXBin(); ir <= fCoord->GetLastXBin(); ++ir) {
      for (Int_t jr = fCoord->GetFirstYBin(); jr <= fCoord->GetLastYBin(); ++jr) {
         for (Int_t kr = fCoord->GetFirstZBin();  kr <= fCoord->GetLastZBin(); ++kr) {
            fMinMaxVal.second = TMath::Max(fMinMaxVal.second, fHist->GetBinContent(ir, jr, kr));
            fMinMaxVal.first = TMath::Min(fMinMaxVal.first, fHist->GetBinContent(ir, jr, kr));
         }
      }
   }

   fXOYSlice.SetMinMax(fMinMaxVal);
   fXOZSlice.SetMinMax(fMinMaxVal);
   fYOZSlice.SetMinMax(fMinMaxVal);

   if (fPolymarker) {
      const Double_t xScale = fCoord->GetXScale();
      const Double_t yScale = fCoord->GetYScale();
      const Double_t zScale = fCoord->GetZScale();

      fPMPoints.assign(fPolymarker->GetP(), fPolymarker->GetP() + fPolymarker->GetN() * 3);
      for (unsigned i = 0; i < fPMPoints.size(); i += 3) {
         fPMPoints[i] *= xScale;
         fPMPoints[i + 1] *= yScale;
         fPMPoints[i + 2] *= zScale;
      }
   }

   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      fXOZSectionPos = fBackBox.Get3DBox()[0].Y();
      fYOZSectionPos = fBackBox.Get3DBox()[0].X();
      fXOYSectionPos = fBackBox.Get3DBox()[0].Z();
      fCoord->ResetModified();
   }

   return kTRUE;
}


//______________________________________________________________________________
void TGLBoxPainter::StartPan(Int_t px, Int_t py)
{
   // User clicks right mouse button (in a pad).

   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}


//______________________________________________________________________________
void TGLBoxPainter::Pan(Int_t px, Int_t py)
{
   // User's moving mouse cursor, with middle mouse button pressed (for pad).
   // Calculate 3d shift related to 2d mouse movement.
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
      //Possibly, move box here
      py = fCamera->GetHeight() - py;
      SaveModelviewMatrix();
      SaveProjectionMatrix();

      fCamera->SetCamera();
      fCamera->Apply(fPadPhi, fPadTheta);


      if (!fHighColor) {
         if (fBoxCut.IsActive() && (fSelectedPart >= kXAxis && fSelectedPart <= kZAxis))
            fBoxCut.MoveBox(px, py, fSelectedPart);
         else
            MoveSection(px, py);
      } else {
         MoveSection(px, py);
      }

      RestoreProjectionMatrix();
      RestoreModelviewMatrix();
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}


//______________________________________________________________________________
void TGLBoxPainter::AddOption(const TString &option)
{
   // Box1 == spheres.
   using namespace std;//isdigit must be in std. But ...

   const Ssiz_t boxPos = option.Index("box");//"box" _already_ _exists_ in a string.
   if (boxPos + 3 < option.Length() && isdigit(option[boxPos + 3]))
      option[boxPos + 3] - '0' == 1 ? fType = kBox1 : fType = kBox;
   else
      fType = kBox;
   option.Index("z") == kNPOS ? fDrawPalette = kFALSE : fDrawPalette = kTRUE;
}

//______________________________________________________________________________
void TGLBoxPainter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{
   // Remove sections.

   if (event == kButton1Double && (HasSections() || fBoxCut.IsActive())) {
      fXOZSectionPos = fBackBox.Get3DBox()[0].Y();
      fYOZSectionPos = fBackBox.Get3DBox()[0].X();
      fXOYSectionPos = fBackBox.Get3DBox()[0].Z();
      if (fBoxCut.IsActive())
         fBoxCut.TurnOnOff();
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLPlotPainter *)0x%lx)->Paint()", (ULong_t)this));
      else
         Paint();
   } else if (event == kKeyPress && (py == kKey_c || py == kKey_C)) {
      if (fHighColor)
         Info("ProcessEvent", "Switch to true color mode to use box cut");
      else {
         fBoxCut.TurnOnOff();
         fUpdateSelection = kTRUE;
      }
   }
}

//______________________________________________________________________________
void TGLBoxPainter::InitGL()const
{
   // Initialize some gl state variables.
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   //For box option back polygons are culled (but not for dynamic profiles).
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGLBoxPainter::DeInitGL()const
{
   //Return back some gl state variables.
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glDisable(GL_LIGHT0);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

namespace {

   //______________________________________________________________________________
   void DrawMinusSigns(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                       Double_t zMin, Double_t zMax, Int_t fp, Bool_t onSphere, Bool_t transp)
   {
      //
      const TGLDisableGuard depthTest(GL_DEPTH_TEST);
      const TGLDisableGuard cullFace(GL_CULL_FACE);

      const Double_t ratio  = onSphere ? 0.4 : 0.15;
      const Double_t leftX = xMin + ratio * (xMax - xMin), rightX = xMax - ratio * (xMax - xMin);
      const Double_t leftY = yMin + ratio * (yMax - yMin), rightY = yMax - ratio * (yMax - yMin);
      const Double_t lowZ = zMin / 2. + zMax / 2. - 0.1 * (zMax - zMin);
      const Double_t upZ = zMin / 2. + zMax / 2. + 0.1 * (zMax - zMin);


      const Double_t minusVerts[][3] = {{xMin, leftY, lowZ}, {xMin, leftY, upZ}, {xMin, rightY, upZ}, {xMin, rightY, lowZ},
                                        {leftX, yMin, lowZ}, {rightX, yMin, lowZ}, {rightX, yMin, upZ}, {leftX, yMin, upZ},
                                        {xMax, leftY, lowZ}, {xMax, rightY, lowZ}, {xMax, rightY, upZ}, {xMax, leftY, upZ},
                                        {rightX, yMax, lowZ}, {leftX, yMax, lowZ}, {leftX, yMax, upZ}, {rightX, yMax, upZ}};
      const Int_t minusQuads[][4] = {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}};


      TGLDisableGuard light(GL_LIGHTING);
      glColor3d(1., 0., 0.);

      const Int_t    frontPlanes[][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};//Code duplication again :(
      const Int_t *verts = minusQuads[frontPlanes[fp][0]];

      glBegin(GL_POLYGON);
      glVertex3dv(minusVerts[verts[0]]);
      glVertex3dv(minusVerts[verts[1]]);
      glVertex3dv(minusVerts[verts[2]]);
      glVertex3dv(minusVerts[verts[3]]);
      glEnd();

      verts = minusQuads[frontPlanes[fp][1]];

      glBegin(GL_POLYGON);
      glVertex3dv(minusVerts[verts[0]]);
      glVertex3dv(minusVerts[verts[1]]);
      glVertex3dv(minusVerts[verts[2]]);
      glVertex3dv(minusVerts[verts[3]]);
      glEnd();

      const Float_t nullEmission[] = {0.f, 0.f, 0.f, 1.f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, nullEmission);
      glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, nullEmission);

      glColor4d(0., 0., 0., 0.25);
      glPolygonMode(GL_FRONT, GL_LINE);

      if (!transp) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      }

      glEnable(GL_LINE_SMOOTH);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

      verts = minusQuads[frontPlanes[fp][0]];

      glBegin(GL_POLYGON);
      glVertex3dv(minusVerts[verts[0]]);
      glVertex3dv(minusVerts[verts[1]]);
      glVertex3dv(minusVerts[verts[2]]);
      glVertex3dv(minusVerts[verts[3]]);
      glEnd();

      verts = minusQuads[frontPlanes[fp][1]];

      glBegin(GL_POLYGON);
      glVertex3dv(minusVerts[verts[0]]);
      glVertex3dv(minusVerts[verts[1]]);
      glVertex3dv(minusVerts[verts[2]]);
      glVertex3dv(minusVerts[verts[3]]);
      glEnd();

      glPolygonMode(GL_FRONT, GL_FILL);

      if (!transp)
         glDisable(GL_BLEND);
   }

}

//______________________________________________________________________________
void TGLBoxPainter::DrawPlot()const
{
   if (fPolymarker)
      return DrawCloud();

   // Draw set of boxes (spheres)

   //Shift plot to point of origin.
   const Rgl::PlotTranslation trGuard(this);

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);
   glDisable(GL_CULL_FACE);
   DrawSections();
   glEnable(GL_CULL_FACE);

   if (!fSelectionPass) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[0
      glPolygonOffset(1.f, 1.f);
      SetPlotColor();
      if (HasSections()) {
         //Boxes are semi-transparent if we have any sections.
         glEnable(GL_BLEND);//[1
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      }
   }

   //Using front point, find the correct order to draw boxes from
   //back to front/from bottom to top (it's important only for semi-transparent boxes).
   const Int_t frontPoint = fBackBox.GetFrontPoint();
   Int_t irInit = fCoord->GetFirstXBin(), iInit = 0;
   const Int_t nX = fCoord->GetNXBins();
   Int_t jrInit = fCoord->GetFirstYBin(), jInit = 0;
   const Int_t nY = fCoord->GetNYBins();
   Int_t krInit = fCoord->GetFirstZBin(), kInit = 0;
   const Int_t nZ = fCoord->GetNZBins();

   const Int_t addI = frontPoint == 2 || frontPoint == 1 ? 1 : (iInit = nX - 1, irInit = fCoord->GetLastXBin(), -1);
   const Int_t addJ = frontPoint == 2 || frontPoint == 3 ? 1 : (jInit = nY - 1, jrInit = fCoord->GetLastYBin(), -1);
   const Int_t addK = fBackBox.Get2DBox()[frontPoint + 4].Y() < fBackBox.Get2DBox()[frontPoint].Y() ? 1
                     : (kInit = nZ - 1, krInit = fCoord->GetLastZBin(),-1);
   const Double_t xScale = fCoord->GetXScale();
   const Double_t yScale = fCoord->GetYScale();
   const Double_t zScale = fCoord->GetZScale();
   const TAxis   *xA = fXAxis;
   const TAxis   *yA = fYAxis;
   const TAxis   *zA = fZAxis;

   if (fSelectionPass && fHighColor)
      Rgl::ObjectIDToColor(fSelectionBase, fHighColor);//base + 1 == 7

   Double_t maxContent = TMath::Max(TMath::Abs(fMinMaxVal.first), TMath::Abs(fMinMaxVal.second));
   if(!maxContent)//bad, find better way to check zero.
      maxContent = 1.;

   for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
      for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
         for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
            const Double_t binContent = fHist->GetBinContent(ir, jr, kr);
            const Double_t w = TMath::Abs(binContent) / maxContent;
            if (!w)
               continue;

            const Double_t xMin = xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 - w * xA->GetBinWidth(ir) / 2);
            const Double_t xMax = xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 + w * xA->GetBinWidth(ir) / 2);
            const Double_t yMin = yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 - w * yA->GetBinWidth(jr) / 2);
            const Double_t yMax = yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 + w * yA->GetBinWidth(jr) / 2);
            const Double_t zMin = zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 - w * zA->GetBinWidth(kr) / 2);
            const Double_t zMax = zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 + w * zA->GetBinWidth(kr) / 2);

            if (fBoxCut.IsActive() && fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            const Int_t binID = fSelectionBase + i * fCoord->GetNZBins() * fCoord->GetNYBins() + j * fCoord->GetNZBins() + k;

            if (fSelectionPass && !fHighColor)
               Rgl::ObjectIDToColor(binID, fHighColor);
            else if(!fHighColor && fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);

            if (fType == kBox) {
               Rgl::DrawBoxFront(xMin, xMax, yMin, yMax, zMin, zMax, frontPoint);
            } else {
               Rgl::DrawSphere(&fQuadric, xMin, xMax, yMin, yMax, zMin, zMax);
            }

            if (binContent < 0. && !fSelectionPass)
               DrawMinusSigns(xMin, xMax, yMin, yMax, zMin, zMax, frontPoint, fType != kBox, HasSections());

            if (!fSelectionPass && !fHighColor && fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);
         }
      }
   }

   if (fBoxCut.IsActive())
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);

   if (!fSelectionPass && fType != kBox1) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      TGLDisableGuard lightGuard(GL_LIGHTING);//[2 - 2]
      glColor4d(0., 0., 0., 0.25);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      const TGLEnableGuard blendGuard(GL_BLEND);//[4-4] + 1]
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      const TGLEnableGuard smoothGuard(GL_LINE_SMOOTH);//[5-5]
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

      for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
         for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
            for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
               const Double_t w = TMath::Abs(fHist->GetBinContent(ir, jr, kr)) / maxContent;
               if (!w)
                  continue;

               const Double_t xMin = xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 - w * xA->GetBinWidth(ir) / 2);
               const Double_t xMax = xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 + w * xA->GetBinWidth(ir) / 2);
               const Double_t yMin = yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 - w * yA->GetBinWidth(jr) / 2);
               const Double_t yMax = yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 + w * yA->GetBinWidth(jr) / 2);
               const Double_t zMin = zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 - w * zA->GetBinWidth(kr) / 2);
               const Double_t zMax = zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 + w * zA->GetBinWidth(kr) / 2);

               if (fBoxCut.IsActive() && fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
                  continue;

               Rgl::DrawBoxFront(xMin, xMax, yMin, yMax, zMin, zMax, frontPoint);
            }
         }
      }

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }

   if (!fSelectionPass && fDrawPalette && HasSections())
      DrawPalette();
}

//______________________________________________________________________________
void TGLBoxPainter::DrawCloud()const
{
   //Draw a frame and a polymarker inside.

   //Shift plot to the point of origin.
   const Rgl::PlotTranslation trGuard(this);

   //Frame.
   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   if (fPhysicalShapeColor)
      glColor3fv(fPhysicalShapeColor);

   glDisable(GL_LIGHTING);

   const TGLVertex3 *bb = fBackBox.Get3DBox();
   const Double_t dX = (bb[1].X() - bb[0].X()) / 40.;
   const Double_t dY = (bb[3].Y() - bb[0].Y()) / 40.;
   const Double_t dZ = (bb[4].Z() - bb[0].Z()) / 40.;
   //Now, draw the cloud of points (polymarker) inside the frame.
   TGLUtil::RenderPolyMarkers(*fPolymarker, fPMPoints, dX, dY, dZ);

   glEnable(GL_LIGHTING);
}

//______________________________________________________________________________
void TGLBoxPainter::SetPlotColor()const
{
   // Set boxes color.

   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.05f};

   if (fPhysicalShapeColor) {
      diffColor[0] = fPhysicalShapeColor[0];
      diffColor[1] = fPhysicalShapeColor[1];
      diffColor[2] = fPhysicalShapeColor[2];
   } else {
      if (fHist->GetFillColor() != kWhite)
         if (const TColor *c = gROOT->GetColor(fHist->GetFillColor()))
            c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   }

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}

//______________________________________________________________________________
void TGLBoxPainter::DrawSectionXOZ()const
{
   // Draw XOZ parallel section.

   if (fSelectionPass)
      return;
   fXOZSlice.DrawSlice(fXOZSectionPos / fCoord->GetYScale());
}

//______________________________________________________________________________
void TGLBoxPainter::DrawSectionYOZ()const
{
   // Draw YOZ parallel section.
   if (fSelectionPass)
      return;
   fYOZSlice.DrawSlice(fYOZSectionPos / fCoord->GetXScale());
}


//______________________________________________________________________________
void TGLBoxPainter::DrawSectionXOY()const
{
   // Draw XOY parallel section.
   if (fSelectionPass)
      return;
   fXOYSlice.DrawSlice(fXOYSectionPos / fCoord->GetZScale());
}


//______________________________________________________________________________
Bool_t TGLBoxPainter::HasSections()const
{
   // Check, if any section exists.

   return fXOZSectionPos > fBackBox.Get3DBox()[0].Y() || fYOZSectionPos> fBackBox.Get3DBox()[0].X() ||
          fXOYSectionPos > fBackBox.Get3DBox()[0].Z();
}

//______________________________________________________________________________
void TGLBoxPainter::DrawPalette()const
{
   //Draw. Palette.
   //Originally, fCamera was never null.
   //It can be a null now because of gl-viewer.
   if (!fCamera) {
      //Thank you, gl-viewer!
      return;
   }

   const TGLLevelPalette * palette = 0;
   const TGLVertex3 *frame = fBackBox.Get3DBox();

   if (fXOZSectionPos > frame[0].Y())
      palette = &fXOZSlice.GetPalette();
   else if (fYOZSectionPos > frame[0].X())
      palette = &fYOZSlice.GetPalette();
   else if (fXOYSectionPos > frame[0].Z())
      palette = &fXOYSlice.GetPalette();

   if (!palette || !palette->GetPaletteSize()) {
      return;
   }

   Rgl::DrawPalette(fCamera, *palette);

   glFinish();

   fCamera->SetCamera();
   fCamera->Apply(fPadPhi, fPadTheta);
}

//______________________________________________________________________________
void TGLBoxPainter::DrawPaletteAxis()const
{
   //Draw. Palette. Axis.
   if (HasSections()) {
      gVirtualX->SetDrawMode(TVirtualX::kCopy);//TCanvas by default sets in kInverse
      Rgl::DrawPaletteAxis(fCamera, fMinMaxVal, fCoord->GetCoordType() == kGLCartesian ? fCoord->GetZLog() : kFALSE);
   }
}
