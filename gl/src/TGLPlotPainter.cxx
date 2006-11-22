// @(#)root/gl:$Name:  $:$Id: TGLPlotPainter.cxx,v 1.7 2006/10/24 14:20:41 brun Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <iostream>
#include <cstdio>

#include "TVirtualPS.h"
#include "TStyle.h"
#include "TError.h"
#include "TAxis.h"
#include "TMath.h"
#include "TH1.h"

#include "TGLPlotPainter.h"
#include "TGLOrthoCamera.h"
#include "TGLIncludes.h"
#include "TGLOutput.h"
#include "gl2ps.h"

ClassImp(TGLPlotPainter)

//______________________________________________________________________________
TGLPlotPainter::TGLPlotPainter(TH1 *hist, TGLOrthoCamera *camera, TGLPlotCoordinates *coord, 
                               Int_t context, Bool_t xoy, Bool_t xoz, Bool_t yoz)
                  : fGLContext(context),
                    fPadColor(0),
                    fHist(hist),
                    fXAxis(hist->GetXaxis()),
                    fYAxis(hist->GetYaxis()),
                    fZAxis(hist->GetZaxis()),
                    fCoord(coord),
                    fCamera(camera),
                    fUpdateSelection(kTRUE),
                    fSelectionPass(kFALSE),
                    fSelectedPart(0),
                    fXOZSectionPos(0.),
                    fYOZSectionPos(0.),
                    fXOYSectionPos(0.),
                    fBackBox(xoy, xoz, yoz),
                    fBoxCut(&fBackBox),
                    fHighColor(kFALSE),
                    fSelectionBase(kTrueColorSelectionBase)
{
   //TGLPlotPainter's ctor.
   if (MakeGLContextCurrent())
      fCamera->SetViewport(GetGLContext());
}

//______________________________________________________________________________
void TGLPlotPainter::Paint()
{
   //Draw lego.
   if (!MakeGLContextCurrent())
      return;

   fHighColor = gGLManager->HighColorFormat(GetGLContext())? kTRUE : kFALSE;
   fSelectionBase = fHighColor ? kHighColorSelectionBase : kTrueColorSelectionBase;

   InitGL();
   //Save material/light properties in a stack.
   glPushAttrib(GL_LIGHTING_BIT);

   fCamera->SetViewport(GetGLContext());
   if (fCamera->ViewportChanged())
      fUpdateSelection = kTRUE;
   //glOrtho etc.
   fCamera->SetCamera();
   //Clear buffer (possibly, with pad's background color).
   ClearBuffers();
   //Set light.
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);
   //Set transformation - shift and rotate the scene.
   fCamera->Apply();
   fBackBox.FindFrontPoint();
   if (gVirtualPS)
      PrintPlot();
   DrawPlot();
   //Restore material properties from stack.
   glPopAttrib();
   glFlush();
   //LegoPainter work is now finished, axes are drawn by axis painter.
   //Here changes are possible in future, if we have real 3d axis painter.
   gGLManager->ReadGLBuffer(GetGLContext());
   //Select pixmap/DIB
   if (fCoord->GetCoordType() == kGLCartesian) {
      gGLManager->SelectOffScreenDevice(GetGLContext());
      //Draw axes into pixmap/DIB
      Int_t viewport[] = {fCamera->GetX(), fCamera->GetY(), fCamera->GetWidth(), fCamera->GetHeight()};
      Rgl::DrawAxes(fBackBox.GetFrontPoint(), viewport, fBackBox.Get2DBox(), fCoord, fXAxis, fYAxis, fZAxis);
   }
   gGLManager->Flush(GetGLContext());
}

//______________________________________________________________________________
void TGLPlotPainter::PrintPlot()const
{
   // Generate PS using gl2ps
   using namespace std;

   TGLOutput::StartEmbeddedPS();
   FILE *output = fopen(gVirtualPS->GetName(), "a");
   Int_t gl2psFormat = GL2PS_EPS;
   Int_t gl2psSort   = GL2PS_BSP_SORT;
   Int_t buffsize    = 0;
   Int_t state       = GL2PS_OVERFLOW;

   while (state == GL2PS_OVERFLOW) {
      buffsize += 1024*1024;
      gl2psBeginPage ("ROOT Scene Graph", "ROOT", NULL,
                      gl2psFormat, gl2psSort, GL2PS_USE_CURRENT_VIEWPORT
                      | GL2PS_POLYGON_OFFSET_FILL | GL2PS_SILENT
                      | GL2PS_BEST_ROOT | GL2PS_OCCLUSION_CULL
                      | 0,
                      GL_RGBA, 0, NULL,0, 0, 0,
                      buffsize, output, NULL);
      DrawPlot();
      state = gl2psEndPage();
   }
   
   fclose(output);
   TGLOutput::CloseEmbeddedPS();
   glFlush();
}

//______________________________________________________________________________
Bool_t TGLPlotPainter::PlotSelected(Int_t px, Int_t py)
{
   // Plot selected.

   if (!MakeGLContextCurrent())
      return kFALSE;
   //Read color buffer content to find selected object
   if (fUpdateSelection) {
      fSelectionPass = kTRUE;
      fCamera->SetCamera();
      TGLDisableGuard lightGuard(GL_LIGHTING);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      fCamera->Apply();
      DrawPlot();
      glFlush();
      fSelection.ReadColorBuffer(fCamera->GetWidth(), fCamera->GetHeight());
      fSelectionPass = kFALSE;
      fUpdateSelection = kFALSE;
   }
   //Convert from window top-bottom into gl bottom-top.
   py = fCamera->GetHeight() - py;
   //Y is a number of a row, x - column.
   std::swap(px, py);
   Int_t newSelected(Rgl::ColorToObjectID(fSelection.GetPixelColor(px, py), fHighColor));

   if (newSelected != fSelectedPart) {
      //New object was selected (or surface deselected) - re-paint.
      fSelectedPart = newSelected;
      gGLManager->MarkForDirectCopy(GetGLContext(), kTRUE);
      Paint();
      gGLManager->MarkForDirectCopy(GetGLContext(), kFALSE);
   }

   return fSelectedPart ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGLPlotPainter::SetGLContext(Int_t context)
{
   //One plot can be painted in several gl contexts.
   fGLContext = context;
}

//______________________________________________________________________________
void TGLPlotPainter::SetPadColor(const TColor *c)
{
   //Used in a pad.
   fPadColor = c;
}

//______________________________________________________________________________
void TGLPlotPainter::SetFrameColor(const TColor *c)
{
   //Set plot's back box color.
   fBackBox.SetFrameColor(c);
}

//______________________________________________________________________________
void TGLPlotPainter::InvalidateSelection()
{
   //Selection must be updated.
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
Int_t TGLPlotPainter::GetGLContext()const
{
   //Get gl context.
   return fGLContext;
}

//______________________________________________________________________________
const TColor *TGLPlotPainter::GetPadColor()const
{
   //Get pad color.
   return fPadColor;
}

//______________________________________________________________________________
Bool_t TGLPlotPainter::MakeGLContextCurrent()const
{
   //Make gl context current.
   return fGLContext != -1 && gGLManager->MakeCurrent(fGLContext);
}

//______________________________________________________________________________
void TGLPlotPainter::MoveSection(Int_t px, Int_t py)
{
   //Create dynamic profile using selected plane
   const TGLVertex3 *frame = fBackBox.Get3DBox();
   const Int_t frontPoint  = fBackBox.GetFrontPoint();

   if (fSelectedPart == 1) {
      fXOYSectionPos = frame[0].Z();
      fSelectedPart = 6;
   } else if (fSelectedPart == 2) {
      if (frontPoint == 2) {
         fXOZSectionPos = frame[0].Y();
         fSelectedPart = 4;
      } else if (!frontPoint) {
         fXOZSectionPos = frame[2].Y();
         fSelectedPart = 4;
      } else if (frontPoint == 1) {
         fYOZSectionPos = frame[0].X();
         fSelectedPart = 5;
      } else if (frontPoint == 3) {
         fYOZSectionPos = frame[1].X();
         fSelectedPart = 5;
      }
   } else if (fSelectedPart == 3) {
      if (frontPoint == 2) {
         fYOZSectionPos = frame[0].X();
         fSelectedPart = 5;
      } else if (!frontPoint) {
         fYOZSectionPos = frame[1].X();
         fSelectedPart = 5;
      } else if (frontPoint == 1) {
         fXOZSectionPos = frame[2].Y();
         fSelectedPart = 4;
      } else if (frontPoint == 3) {
         fXOZSectionPos = frame[0].Y();
         fSelectedPart = 4;
      }
   }

   Double_t mv[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mv);
   Double_t pr[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, pr);
   Int_t vp[4] = {0};
   glGetIntegerv(GL_VIEWPORT, vp);
   Double_t winVertex[3] = {0.};

   if (fSelectedPart == 6)
      gluProject(0., 0., fXOYSectionPos, mv, pr, vp, &winVertex[0], &winVertex[1], &winVertex[2]);
   else
      gluProject(fSelectedPart == 5 ? fYOZSectionPos : 0., 
                 fSelectedPart == 4 ? fXOZSectionPos : 0., 
                 0., mv, pr, vp, 
                 &winVertex[0], &winVertex[1], &winVertex[2]);
   winVertex[0] += px - fMousePosition.fX;
   winVertex[1] += py - fMousePosition.fY;
   Double_t newPoint[3] = {0.};
   gluUnProject(winVertex[0], winVertex[1], winVertex[2], mv, pr, vp,
                newPoint, newPoint + 1, newPoint + 2);

   if (fSelectedPart == 4)
      fXOZSectionPos = newPoint[1];
   else if (fSelectedPart == 5)
      fYOZSectionPos = newPoint[0];
   else
      fXOYSectionPos = newPoint[2];
}

//______________________________________________________________________________
void TGLPlotPainter::DrawSections()const
{
   //Draw sections (if any).
   const TGLVertex3 *frame = fBackBox.Get3DBox();
   
   if (fXOZSectionPos > frame[0].Y()) {
      if (fXOZSectionPos > frame[2].Y())
         fXOZSectionPos = frame[2].Y();
      const TGLVertex3 v1(frame[0].X(), fXOZSectionPos, frame[0].Z());
      const TGLVertex3 v2(frame[4].X(), fXOZSectionPos, frame[4].Z());
      const TGLVertex3 v3(frame[5].X(), fXOZSectionPos, frame[5].Z());
      const TGLVertex3 v4(frame[1].X(), fXOZSectionPos, frame[1].Z());

      if (fSelectionPass)
         Rgl::ObjectIDToColor(4, fHighColor);
      else if (fSelectedPart == 4)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gBlueEmission);
  
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
      Rgl::DrawQuadFilled(v1, v2, v3, v4, TGLVector3(0., 1., 0.));
      glDisable(GL_POLYGON_OFFSET_FILL);
      //Zlevels here.
      if (!fSelectionPass) {
         if (fSelectedPart == 4)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
         const TGLDisableGuard lightGuard(GL_LIGHTING);
         const TGLEnableGuard  blendGuard(GL_BLEND);
         const TGLEnableGuard  lineSmooth(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
         glDepthMask(GL_FALSE);
         DrawSectionXOZ();
         //Draw z-levels
         const TGLEnableGuard stippleGuard(GL_LINE_STIPPLE);//[1-1]
         const UShort_t stipple = 0x5555;
         glLineStipple(1, stipple);

         glColor3d(0., 0., 0.);
         glBegin(GL_LINES);
         for (UInt_t i = 0; i < fZLevels.size(); ++i) {
            glVertex3d(fBackBox.Get3DBox()[1].X(), fXOZSectionPos, fZLevels[i]);
            glVertex3d(fBackBox.Get3DBox()[0].X(), fXOZSectionPos, fZLevels[i]);
         }
         glEnd();
         glDepthMask(GL_TRUE);
      }
   }

   if (fYOZSectionPos > frame[0].X()) {
      if (fYOZSectionPos > frame[1].X())
         fYOZSectionPos = frame[1].X();
      TGLVertex3 v1(fYOZSectionPos, frame[0].Y(), frame[0].Z());
      TGLVertex3 v2(fYOZSectionPos, frame[3].Y(), frame[3].Z());
      TGLVertex3 v3(fYOZSectionPos, frame[7].Y(), frame[7].Z());
      TGLVertex3 v4(fYOZSectionPos, frame[4].Y(), frame[4].Z());
      
      if (fSelectionPass) {
         Rgl::ObjectIDToColor(5, fHighColor);
      } else if (fSelectedPart == 5)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gBlueEmission);
      
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
      Rgl::DrawQuadFilled(v1, v2, v3, v4, TGLVector3(1., 0., 0.));
      glDisable(GL_POLYGON_OFFSET_FILL);

      if (!fSelectionPass) {
         if (fSelectedPart == 5)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
         const TGLDisableGuard lightHuard(GL_LIGHTING);
         const TGLEnableGuard blendGuard(GL_BLEND);
         const TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
         glDepthMask(GL_FALSE);
         DrawSectionYOZ();
         //Draw z-levels
         const TGLEnableGuard stippleGuard(GL_LINE_STIPPLE);//[1-1]
         glLineStipple(1, 0x5555);

         glColor3d(0., 0., 0.);
         glBegin(GL_LINES);
         for (UInt_t i = 0; i < fZLevels.size(); ++i) {
            glVertex3d(fYOZSectionPos, fBackBox.Get3DBox()[3].Y(), fZLevels[i]);
            glVertex3d(fYOZSectionPos, fBackBox.Get3DBox()[0].Y(), fZLevels[i]);
         }
         glEnd();
         glDepthMask(GL_TRUE);
      }
   }

   if (fXOYSectionPos > frame[0].Z()) {
      if (fXOYSectionPos > frame[4].Z())
         fXOYSectionPos = frame[4].Z();
      TGLVertex3 v1(frame[0].X(), frame[0].Y(), fXOYSectionPos);
      TGLVertex3 v2(frame[1].X(), frame[1].Y(), fXOYSectionPos);
      TGLVertex3 v3(frame[2].X(), frame[2].Y(), fXOYSectionPos);
      TGLVertex3 v4(frame[3].X(), frame[3].Y(), fXOYSectionPos);
      
      if (fSelectionPass) {
         Rgl::ObjectIDToColor(6, fHighColor);
      } else if (fSelectedPart == 6)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gBlueEmission);
      
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
      //if (fSelectionPass || fSelectedPart == 6)
      Rgl::DrawQuadFilled(v1, v2, v3, v4, TGLVector3(0., 0., 1.));
      glDisable(GL_POLYGON_OFFSET_FILL);

      if (!fSelectionPass) {
         if (fSelectedPart == 6)
            glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);
         const TGLDisableGuard lightGuard(GL_LIGHTING);
         const TGLEnableGuard blendGuard(GL_BLEND);
         const TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
         glDepthMask(GL_FALSE);
         DrawSectionXOY();
         glDepthMask(GL_TRUE);
      }
   }
}

ClassImp(TGLPlotCoordinates)

//______________________________________________________________________________
TGLPlotCoordinates::TGLPlotCoordinates()
                        : fCoordType(kGLCartesian),
                          fXScale(1.),
                          fYScale(1.),
                          fZScale(1.),
                          fXLog(kFALSE),
                          fYLog(kFALSE),
                          fZLog(kFALSE),
                          fModified(kFALSE)
{
   //Constructor.
}

//______________________________________________________________________________
TGLPlotCoordinates::~TGLPlotCoordinates()
{
   //Dtor.
}

//______________________________________________________________________________
void TGLPlotCoordinates::SetCoordType(EGLCoordType type)
{
   //If coord type was changed, plot must reset sections (if any),
   //set fModified.
   if (fCoordType != type) {
      fModified = kTRUE;
      fCoordType = type;
   }
}

//______________________________________________________________________________
EGLCoordType TGLPlotCoordinates::GetCoordType()const
{
   // Get coordinates type.

   return fCoordType;
}

//______________________________________________________________________________
void TGLPlotCoordinates::SetXLog(Bool_t xLog)
{
   //If log changed, sections must be reset, 
   //set fModified.
   if (fXLog != xLog) {
      fXLog = xLog;
      fModified = kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::GetXLog()const
{
   // Get X log.

   return fXLog;
}

//______________________________________________________________________________
void TGLPlotCoordinates::SetYLog(Bool_t yLog)
{
   //If log changed, sections must be reset, 
   //set fModified.
   if (fYLog != yLog) {
      fYLog = yLog;
      fModified = kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::GetYLog()const
{
   // Get Y log.

   return fYLog;
}

//______________________________________________________________________________
void TGLPlotCoordinates::SetZLog(Bool_t zLog)
{
   //If log changed, sections must be reset, 
   //set fModified.
   if (fZLog != zLog) {
      fZLog = zLog;
      fModified = kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::GetZLog()const
{
   // Get Z log.

   return fZLog;
}

//______________________________________________________________________________
void TGLPlotCoordinates::ResetModified()
{
   // Reset modified.

   fModified = kFALSE;
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::Modified()const
{
   // Modified.

   return fModified;
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::SetRanges(const TH1 *hist, Bool_t errors, Bool_t zBins)
{
   //Set bin ranges, ranges.
   switch (fCoordType) {
   case kGLPolar:
      return SetRangesPolar(hist);
   case kGLCylindrical:
      return SetRangesCylindrical(hist);
   case kGLSpherical:
      return SetRangesSpherical(hist);
   case kGLCartesian:
   default:
      return SetRangesCartesian(hist, errors, zBins);
   }
}

//______________________________________________________________________________
Int_t TGLPlotCoordinates::GetNXBins()const
{
   //Number of X bins.
   return fXBins.second - fXBins.first + 1;
}

//______________________________________________________________________________
Int_t TGLPlotCoordinates::GetNYBins()const
{
   //Number of Y bins.
   return fYBins.second - fYBins.first + 1;
}

//______________________________________________________________________________
Int_t TGLPlotCoordinates::GetNZBins()const
{
   //Number of Z bins.
   return fZBins.second - fZBins.first + 1;
}

//______________________________________________________________________________
const Rgl::BinRange_t &TGLPlotCoordinates::GetXBins()const
{
   //X bins range.
   return fXBins;
}

//______________________________________________________________________________
const Rgl::BinRange_t &TGLPlotCoordinates::GetYBins()const
{
   //Y bins range.
   return fYBins;
}

//______________________________________________________________________________
const Rgl::BinRange_t &TGLPlotCoordinates::GetZBins()const
{
   //Z bins range.
   return fZBins;
}

//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetXRange()const
{
   //X range.
   return fXRange;
}

//______________________________________________________________________________
Double_t TGLPlotCoordinates::GetXLength()const
{
   //X length.
   return fXRange.second - fXRange.first;
}

//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetYRange()const
{
   //Y range.
   return fYRange;
}

//______________________________________________________________________________
Double_t TGLPlotCoordinates::GetYLength()const
{
   //Y length.
   return fYRange.second - fYRange.first;
}


//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetZRange()const
{
   //Z range.
   return fZRange;
}

//______________________________________________________________________________
Double_t TGLPlotCoordinates::GetZLength()const
{
   //Z length.
   return fZRange.second - fZRange.first;
}


//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetXRangeScaled()const
{
   //Scaled range.
   return fXRangeScaled;
}

//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetYRangeScaled()const
{
   //Scaled range.
   return fYRangeScaled;
}

//______________________________________________________________________________
const Rgl::Range_t &TGLPlotCoordinates::GetZRangeScaled()const
{
   //Scaled range.
   return fZRangeScaled;
}

//______________________________________________________________________________
Double_t TGLPlotCoordinates::GetFactor()const
{
   // Get factor.

   return fFactor;
}

namespace {

   Bool_t FindAxisRange(const TAxis *axis, Bool_t log, Rgl::BinRange_t &bins, Rgl::Range_t &range);
   Bool_t FindAxisRange(const TH1 *hist, Bool_t logZ, const Rgl::BinRange_t &xBins, 
                        const Rgl::BinRange_t &yBins, Rgl::Range_t &zRange, 
                        Double_t &factor, Bool_t errors);

}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::SetRangesCartesian(const TH1 *hist, Bool_t errors, Bool_t zAsBins)
{
   //Set bin ranges, ranges, etc.
   Rgl::BinRange_t xBins;
   Rgl::Range_t    xRange;
   const TAxis *xAxis = hist->GetXaxis();
   if (!FindAxisRange(xAxis, fXLog, xBins, xRange)) {
      Error("TGLPlotCoordinates::SetRangesCartesian", "Cannot set X axis to log scale");
      return kFALSE;
   }

   Rgl::BinRange_t yBins;
   Rgl::Range_t    yRange;
   const TAxis *yAxis = hist->GetYaxis();
   if (!FindAxisRange(yAxis, fYLog, yBins, yRange)) {
      Error("TGLPlotCoordinates::SetRangesCartesian", "Cannot set Y axis to log scale");
      return kFALSE;
   }

   Rgl::BinRange_t zBins;
   Rgl::Range_t zRange;
   Double_t factor = 1.;

   if (zAsBins) {
      if (!FindAxisRange(hist->GetZaxis(), fZLog, zBins, zRange)) {
         Error("TGLPlotCoordinates::SetRangesCartesian", "Cannot set Z axis to log scale");
         return kFALSE;
      }
   } else if (!FindAxisRange(hist, fZLog, xBins, yBins, zRange, factor, errors)) {
      Error("TGLPlotCoordinates::SetRangesCartesian", 
            "Log scale is requested for Z, but maximum less or equal 0. (%f)", zRange.second);
      return kFALSE;
   }

   //Finds the maximum dimension and adjust scale coefficients
   const Double_t x = xRange.second - xRange.first;
   const Double_t y = yRange.second - yRange.first;
   const Double_t z = zRange.second - zRange.first;

   if (!x || !y || !z) {
      Error("TGLPlotCoordinates::SetRangesCartesian", "Zero axis range.");
      return kFALSE;
   }

   if (xRange != fXRange || yRange != fYRange || zRange != fZRange ||
       xBins != fXBins || yBins != fYBins || zBins != fZBins || fFactor != factor)
   {
      fModified = kTRUE;
   }

   fXRange = xRange, fXBins = xBins, fYRange = yRange, fYBins = yBins, fZRange = zRange, fZBins = zBins;
   fFactor = factor;

   const Double_t maxDim = TMath::Max(TMath::Max(x, y), z);
   fXScale = maxDim / x;
   fYScale = maxDim / y;
   fZScale = maxDim / z;
   fXRangeScaled.first = fXRange.first * fXScale, fXRangeScaled.second = fXRange.second * fXScale;
   fYRangeScaled.first = fYRange.first * fYScale, fYRangeScaled.second = fYRange.second * fYScale;
   fZRangeScaled.first = fZRange.first * fZScale, fZRangeScaled.second = fZRange.second * fZScale;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::SetRangesPolar(const TH1 *hist)
{
   //Set bin ranges, ranges, etc.
   Rgl::BinRange_t xBins;
   Rgl::Range_t phiRange;
   const TAxis *xAxis = hist->GetXaxis();
   FindAxisRange(xAxis, kFALSE, xBins, phiRange);
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLPlotCoordinates::SetRangesPolar", "To many PHI sectors");
      return kFALSE;
   }

   Rgl::BinRange_t yBins;
   Rgl::Range_t roRange;
   const TAxis *yAxis = hist->GetYaxis();
   FindAxisRange(yAxis, kFALSE, yBins, roRange);

   Rgl::Range_t zRange;
   Double_t factor = 1.;
   if (!FindAxisRange(hist, fZLog, xBins, yBins, zRange, factor, kFALSE))
   {
      Error("TGLPlotCoordinates::SetRangesPolar", 
            "Log scale is requested for Z, but maximum less or equal 0. (%f)", zRange.second);
      return kFALSE;
   }

   const Double_t z = zRange.second - zRange.first;
   if (!z || !(phiRange.second - phiRange.first) || !(roRange.second - roRange.first)) {
      Error("TGLPlotCoordinates::SetRangesPolar", "Zero axis range.");
      return kFALSE;
   }

   if (phiRange != fXRange || roRange != fYRange || zRange != fZRange ||
       xBins != fXBins || yBins != fYBins || fFactor != factor)
   {
      fModified = kTRUE;
      fXRange = phiRange, fXBins = xBins;
      fYRange = roRange,  fYBins = yBins;
      fZRange = zRange;
      fFactor = factor;
   }

   const Double_t maxDim = TMath::Max(2., z);
   fXScale = maxDim / 2.;
   fYScale = maxDim / 2.;
   fZScale = maxDim / z;
   fXRangeScaled.first = -fXScale, fXRangeScaled.second = fXScale;
   fYRangeScaled.first = -fYScale, fYRangeScaled.second = fYScale;
   fZRangeScaled.first = fZRange.first * fZScale, fZRangeScaled.second = fZRange.second * fZScale;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::SetRangesCylindrical(const TH1 *hist)
{
   // Set ranges cylindrical.

   Rgl::BinRange_t xBins, yBins;
   Rgl::Range_t angleRange, heightRange, radiusRange;
   const TAxis *xAxis = hist->GetXaxis();
   const TAxis *yAxis = hist->GetYaxis();
   Double_t factor = 1.;

   FindAxisRange(xAxis, kFALSE, xBins, angleRange);
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLPlotCoordinates::SetRangesCylindrical", "To many PHI sectors");
      return kFALSE;
   }
   if (!FindAxisRange(yAxis, fYLog, yBins, heightRange)) {
      Error("TGLPlotCoordinates::SetRangesCylindrical", "Cannot set Y axis to log scale");
      return kFALSE;
   }
   FindAxisRange(hist, kFALSE, xBins, yBins, radiusRange, factor, kFALSE);

   const Double_t x = angleRange.second  - angleRange.first;
   const Double_t y = heightRange.second - heightRange.first;
   const Double_t z = radiusRange.second - radiusRange.first;

   if (!x || !y || !z) {
      Error("TGLPlotCoordinates::SetRangesCylindrical", "Zero axis range.");
      return kFALSE;
   }

   if (angleRange != fXRange  || heightRange != fYRange || 
       radiusRange != fZRange || xBins != fXBins || 
       yBins != fYBins || fFactor != factor) 
   {
      fModified = kTRUE;
      fXRange = angleRange,  fXBins = xBins;
      fYRange = heightRange, fYBins = yBins;
      fZRange = radiusRange;
      fFactor = factor;
   }

   const Double_t maxDim = TMath::Max(2., y);
   fXScale = maxDim / 2.;
   fYScale = maxDim / y;
   fZScale = maxDim / 2.;
   fXRangeScaled.first = -fXScale, fXRangeScaled.second = fXScale;
   fYRangeScaled.first = fYRange.first * fYScale, fYRangeScaled.second = fYRange.second * fYScale;
   fZRangeScaled.first = -fZScale, fZRangeScaled.second = fZScale;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLPlotCoordinates::SetRangesSpherical(const TH1 *hist)
{
   // Set ranges spherical.

   Rgl::BinRange_t xBins;
   Rgl::Range_t phiRange;
   FindAxisRange(hist->GetXaxis(), kFALSE, xBins, phiRange);
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLPlotCoordinates::SetRangesSpherical", "To many PHI sectors");
      return kFALSE;
   }

   Rgl::BinRange_t yBins;
   Rgl::Range_t thetaRange;
   FindAxisRange(hist->GetYaxis(), kFALSE, yBins, thetaRange);
   if (yBins.second - yBins.first + 1 > 180) {
      Error("TGLPlotCoordinates::SetRangesSpherical", "To many THETA sectors");
      return kFALSE;
   }

   Rgl::Range_t radiusRange;
   Double_t factor = 1.;
   FindAxisRange(hist, kFALSE, xBins, yBins, radiusRange, factor, kFALSE);

   if (xBins != fXBins || yBins != fYBins || 
       phiRange != fXRange || thetaRange != fYRange || 
       radiusRange != fZRange || fFactor != factor)
   {
      fModified = kTRUE;
      fXBins    = xBins;
      fYBins    = yBins;
      fXRange   = phiRange;
      fYRange   = thetaRange, 
      fZRange   = radiusRange;
      fFactor   = factor;
   }

   fXScale = 1.;
   fYScale = 1.;
   fZScale = 1.;
   fXRangeScaled.first = -fXScale, fXRangeScaled.second = fXScale;
   fYRangeScaled.first = -fYScale, fYRangeScaled.second = fYScale;
   fZRangeScaled.first = -fZScale, fZRangeScaled.second = fZScale;

   return kTRUE;
}

namespace {

   //______________________________________________________________________________
   Double_t FindMinBinWidth(const TAxis *axis)
   {
      // Find minimal bin width.

      Int_t currBin = axis->GetFirst();
      Double_t width = axis->GetBinWidth(currBin);

      if (!axis->IsVariableBinSize())//equal bins
         return width;

      ++currBin;
      //variable size bins
      for (const Int_t lastBin = axis->GetLast(); currBin <= lastBin; ++currBin)
         width = TMath::Min(width, axis->GetBinWidth(currBin));

      return width;
   }

   //______________________________________________________________________________
   Bool_t FindAxisRange(const TAxis *axis, Bool_t log, Rgl::BinRange_t &bins, Rgl::Range_t &range)
   {
      //"Generic" function, can be used for X/Y/Z axis.
      //[low edge of first ..... up edge of last]
      //If log is true, at least up edge of last MUST be positive or function fails (1).
      //If log is true and low edge is negative, try to find bin with positive low edge, bin number
      //must be less or equal to last (2). If no such bin, function failes.
      //When looking for a such bin, I'm trying to find value which is 0.01 of
      //MINIMUM bin width (3) (if bins are equidimensional, first's bin width is OK).
      //But even such lookup can fail, so, it's a stupid idea to have negative ranges
      //and logarithmic scale :)

      bins.first = axis->GetFirst(), bins.second = axis->GetLast();
      range.first = axis->GetBinLowEdge(bins.first), range.second = axis->GetBinUpEdge(bins.second);

      if (log) {
         if (range.second <= 0.)
            return kFALSE;//(1)

         range.second = TMath::Log10(range.second);

         if (range.first <= 0.) {//(2)
            Int_t bin = axis->FindFixBin(FindMinBinWidth(axis) * 0.01);//(3)
            //Overflow or something stupid.
            if (bin > bins.second)
               return kFALSE;
            
            if (axis->GetBinLowEdge(bin) <= 0.) {
               ++bin;
               if (bin > bins.second)//Again, something stupid.
                  return kFALSE;
            }

            bins.first = bin;
            range.first = axis->GetBinLowEdge(bin);
         }

         range.first = TMath::Log10(range.first);
      }

      return kTRUE;
   }

   //______________________________________________________________________________
   Bool_t FindAxisRange(const TH1 *hist, Bool_t logZ, const Rgl::BinRange_t &xBins, 
                        const Rgl::BinRange_t &yBins, Rgl::Range_t &zRange, 
                        Double_t &factor, Bool_t errors)
   {
      //First, look through hist to find minimum and maximum values.
      const Bool_t minimum = hist->GetMinimumStored() != -1111;
      const Bool_t maximum = hist->GetMaximumStored() != -1111;
      const Double_t margin = gStyle->GetHistTopMargin();

      zRange.second = hist->GetCellContent(xBins.first, yBins.first), zRange.first = zRange.second;
      Double_t summ = 0.;

      for (Int_t i = xBins.first; i <= xBins.second; ++i) {
         for (Int_t j = yBins.first; j <= yBins.second; ++j) {
            Double_t val = hist->GetCellContent(i, j);
            if (val > 0. && errors)
               val = TMath::Max(val, val + hist->GetCellError(i, j));
            zRange.second = TMath::Max(val, zRange.second);
            zRange.first = TMath::Min(val, zRange.first);
            summ += val;
         }
      }

      if (hist->GetMaximumStored() != -1111) 
         zRange.second = hist->GetMaximumStored();
      if (hist->GetMinimumStored() != -1111) 
         zRange.first = hist->GetMinimumStored();

      if (logZ && zRange.second <= 0.)
         return kFALSE;//cannot setup logarithmic scale
      
      if (zRange.first >= zRange.second)
         zRange.first = 0.001 * zRange.second;

      factor = hist->GetNormFactor() > 0. ? hist->GetNormFactor() : summ;
      if (summ) factor /= summ;
      if (!factor) factor = 1.;
      if (factor < 0.)
         Warning("TGLPlotPainter::ExtractAxisZInfo", 
               "Negative factor, negative ranges - possible incorrect behavior");

      zRange.second *= factor;
      zRange.first  *= factor;

      if (logZ) {
         if (zRange.first <= 0.)
            zRange.first = TMath::Min(1., 0.001 * zRange.second);
         zRange.first = TMath::Log10(zRange.first);
         if (!minimum) 
            zRange.first += TMath::Log10(0.5);
         zRange.second = TMath::Log10(zRange.second);
         if (!maximum)
            zRange.second += TMath::Log10(2*(0.9/0.95));//This magic numbers are from THistPainter.
         return kTRUE;
      }

      if (!maximum)
         zRange.second += margin * (zRange.second - zRange.first);
      if (!minimum) {
         if (gStyle->GetHistMinimumZero())
            zRange.first >= 0 ? zRange.first = 0. : zRange.first -= margin * (zRange.second - zRange.first);
         else 
            zRange.first >= 0 && zRange.first - margin * (zRange.second - zRange.first) <= 0 ?
               zRange.first = 0 : zRange.first -= margin * (zRange.second - zRange.first);
      }

      return kTRUE;
   }

}

ClassImp(TGLBoxCut)

//______________________________________________________________________________
TGLBoxCut::TGLBoxCut(const TGLPlotBox *box)
               : fDirection(kAlongX),
                 fXLength(0.),
                 fYLength(0.),
                 fZLength(0.),
                 fPlotBox(box),
                 fActive(kFALSE),
                 fFactor(1.)
{
   //Constructor.
}

//______________________________________________________________________________
TGLBoxCut::~TGLBoxCut()
{
   //Destructor.
}

//______________________________________________________________________________
void TGLBoxCut::TurnOnOff()
{
   fActive = !fActive;

   if (fActive) {
      const Int_t frontPoint = fPlotBox->GetFrontPoint();
      const TGLVertex3 *box = fPlotBox->Get3DBox();
      const TGLVertex3 center((box[0].X() + box[1].X()) / 2, (box[0].Y() + box[2].Y()) / 2,
                              (box[0].Z() + box[4].Z()) / 2);
      fXLength = fFactor * (box[1].X() - box[0].X());
      fYLength = fFactor * (box[2].Y() - box[0].Y());
      fZLength = fFactor * (box[4].Z() - box[0].Z());

      switch(frontPoint){
      case 0:
         fCenter.X() = box[0].X();
         fCenter.Y() = box[0].Y();
         break;
      case 1:
         fCenter.X() = box[1].X();
         fCenter.Y() = box[0].Y();
         break;
      case 2:
         fCenter.X() = box[2].X();
         fCenter.Y() = box[2].Y();
         break;
      case 3:
         fCenter.X() = box[0].X();
         fCenter.Y() = box[2].Y();
         break;
      }

      fCenter.Z() = box[4].Z();
      AdjustBox();
   }
}

void TGLBoxCut::DrawBox(Bool_t selectionPass, Int_t selected)const
{
   if (!selectionPass) {
      GLboolean oldBlendState = kFALSE;
      glGetBooleanv(GL_BLEND, &oldBlendState);
   
      if (!oldBlendState)
         glEnable(GL_BLEND);
      
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

      glDisable(GL_LIGHTING);
      glLineWidth(3.f);
      glEnable(GL_LINE_SMOOTH);
      glColor4d(1., 0.3, 0., 0.8);
      switch(fDirection){
      case kAlongX:
         glBegin(GL_LINES);
         glVertex3d(fXRange.first, (fYRange.first + fYRange.second) / 2, (fZRange.first + fZRange.second) / 2);
         glVertex3d(fXRange.second, (fYRange.first + fYRange.second) / 2, (fZRange.first + fZRange.second) / 2);
         glEnd();
         break;
      case kAlongY:
         glBegin(GL_LINES);
         glVertex3d((fXRange.first + fXRange.second) / 2, fYRange.first, (fZRange.first + fZRange.second) / 2);
         glVertex3d((fXRange.first + fXRange.second) / 2, fYRange.second, (fZRange.first + fZRange.second) / 2);
         glEnd();
         break;
      case kAlongZ:
         glBegin(GL_LINES);
         glVertex3d((fXRange.first + fXRange.second) / 2, (fYRange.first + fYRange.second) / 2, fZRange.first);
         glVertex3d((fXRange.first + fXRange.second) / 2, (fYRange.first + fYRange.second) / 2, fZRange.second);
         glEnd();
         break;
      }

      glDisable(GL_LINE_SMOOTH);
      glLineWidth(1.f);
      glEnable(GL_LIGHTING);

      const Float_t diffuseColor[] = {0.f, 0.f, 1.f, selected == 7 ? 0.5f : 0.2f};
      glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuseColor);
   
      if (selected == 7) {
         const Float_t blueEmission[] = {0.5f, 0.f, 1.f, 1.f};
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, blueEmission);
      }

      Rgl::DrawBoxFront(fXRange.first, fXRange.second, fYRange.first, fYRange.second,
                        fZRange.first, fZRange.second, fPlotBox->GetFrontPoint());

      if (selected == 7);
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, Rgl::gNullEmission);

      if (!oldBlendState)
         glDisable(GL_BLEND);
   } else {
      Rgl::ObjectIDToColor(7, kFALSE);//kFALSE == highColor
      Rgl::DrawBoxFront(fXRange.first, fXRange.second, fYRange.first, fYRange.second,
                        fZRange.first, fZRange.second, fPlotBox->GetFrontPoint());
   }
}

void TGLBoxCut::SetDirectionX()
{
   fDirection = kAlongX;
   //
}

void TGLBoxCut::SetDirectionY()
{
   fDirection = kAlongY;
   //
}

void TGLBoxCut::SetDirectionZ()
{
   fDirection = kAlongZ;
}

void TGLBoxCut::StartMovement(Int_t px, Int_t py)
{
   fMousePos.fX = px;
   fMousePos.fY = py;
}

void TGLBoxCut::MoveBox(Int_t px, Int_t py)
{
   Double_t mv[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mv);
   Double_t pr[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, pr);
   Int_t vp[4] = {0};
   glGetIntegerv(GL_VIEWPORT, vp);
   Double_t winVertex[3] = {0.};

   switch(fDirection){
   case kAlongX:
      gluProject(fCenter.X(), 0., 0., mv, pr, vp, &winVertex[0], &winVertex[1], &winVertex[2]);
      break;
   case kAlongY:
      gluProject(0., fCenter.Y(), 0., mv, pr, vp, &winVertex[0], &winVertex[1], &winVertex[2]);
      break;
   case kAlongZ:
      gluProject(0., 0., fCenter.Z(), mv, pr, vp, &winVertex[0], &winVertex[1], &winVertex[2]);
      break;
   }

   winVertex[0] += px - fMousePos.fX;
   winVertex[1] += py - fMousePos.fY;
   Double_t newPoint[3] = {0.};
   gluUnProject(winVertex[0], winVertex[1], winVertex[2], mv, pr, vp,
                newPoint, newPoint + 1, newPoint + 2);

   switch(fDirection){
   case kAlongX:
      fCenter.X() = newPoint[0];
      break;
   case kAlongY:
      fCenter.Y() = newPoint[1];
      break;
   case kAlongZ:
      fCenter.Z() = newPoint[2];
      break;
   }
   
   fMousePos.fX = px;
   fMousePos.fY = py;
   
   AdjustBox();
}

void TGLBoxCut::AdjustBox()
{
   const TGLVertex3 *box = fPlotBox->Get3DBox();
   
   fXRange.first  = fCenter.X() - fXLength / 2.;
   fXRange.second = fCenter.X() + fXLength / 2.;
   fYRange.first  = fCenter.Y() - fYLength / 2.;
   fYRange.second = fCenter.Y() + fYLength / 2.;
   fZRange.first  = fCenter.Z() - fZLength / 2.;
   fZRange.second = fCenter.Z() + fZLength / 2.;

   fXRange.first  = TMath::Max(fXRange.first, box[0].X());
   fXRange.first  = TMath::Min(fXRange.first, box[1].X());
   fXRange.second = TMath::Min(fXRange.second, box[1].X());
   fXRange.second = TMath::Max(fXRange.second, box[0].X());
   
   fYRange.first  = TMath::Max(fYRange.first, box[0].Y());
   fYRange.first  = TMath::Min(fYRange.first, box[2].Y());
   fYRange.second = TMath::Min(fYRange.second, box[2].Y());
   fYRange.second = TMath::Max(fYRange.second, box[0].Y());

   fZRange.first  = TMath::Max(fZRange.first, box[0].Z());
   fZRange.first  = TMath::Min(fZRange.first, box[4].Z());
   fZRange.second = TMath::Min(fZRange.second, box[4].Z());
   fZRange.second = TMath::Max(fZRange.second, box[0].Z());
   
   if (fXRange.second - fXRange.first < 0.001 || 
       fYRange.second - fYRange.first < 0.001 || 
       fZRange.second - fZRange.first < 0.001) {
      fActive = kFALSE;
   }
}

Bool_t TGLBoxCut::IsInCut(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax,
                          Double_t zMin, Double_t zMax)const
{
   if (((xMin >= fXRange.first && xMin < fXRange.second) || (xMax > fXRange.first && xMax <= fXRange.second)) &&
       ((yMin >= fYRange.first && yMin < fYRange.second) || (yMax > fYRange.first && yMax <= fYRange.second)) &&
       ((zMin >= fZRange.first && zMin < fZRange.second) || (zMax > fZRange.first && zMax <= fZRange.second)))
       return kTRUE;
   return kFALSE;
}
