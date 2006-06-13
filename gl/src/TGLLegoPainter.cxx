#include <algorithm>
#include <cctype>

#include "Buttons.h"
#include "TString.h"
#include "TStyle.h"
#include "TColor.h"
#include "TError.h"
#include "TMath.h"
#include "TAxis.h"
#include "TH1.h"

#include "TGLLegoPainter.h"
#include "TGLAxisPainter.h"
#include "TGLIncludes.h"

ClassImp(TGLLegoPainter)

const Float_t   TGLLegoPainter::fRedEmission[] = {1.f, 0.4f, 0.f, 1.f};
const Float_t  TGLLegoPainter::fNullEmission[] = {0.f, 0.f,  0.f, 1.f};
const Float_t TGLLegoPainter::fGreenEmission[] = {0.f, 1.f,  0.f, 1.f};

namespace {
   //Must be in TGLUtil.h!
   const UChar_t gDefTexture[] =
   {
      //R    G    B    A
      128, 0,   255, 200,
      169, 4,   240, 200,
      199, 73,  255, 200,
      222, 149, 253, 200,
      255, 147, 201, 200,
      255, 47,  151, 200,
      232, 0,   116, 200,
      253, 0,   0,   200,
      255, 62,  62,  200,
      217, 111, 15,  200,
      242, 151, 28,  200,
      245, 172, 73,  200,
      251, 205, 68,  200,
      255, 255, 21,  200,
      255, 255, 128, 200,
      255, 255, 185, 200
   };
}

//______________________________________________________________________________
TGLLegoPainter::TGLLegoPainter(TH1 *hist, TGLAxisPainter *axisPainter, Int_t ctx, EGLCoordType type, 
                               Bool_t logX, Bool_t logY, Bool_t logZ)
                  : TGLPlotFrame(logX, logY, logZ),
                    fHist(hist),
                    fGLContext(ctx),
                    fCoordType(type),
                    fMinZ(0.),
                    fLegoType(kColorSimple),
                    fSelectionPass(kFALSE),
                    fUpdateSelection(kTRUE),
                    fSelectedBin(-1, -1),
                    fSelectionMode(kSelectionSimple),
                    fSelectedPlane(0),
                    fPadColor(0),
                    fFrameColor(0),
                    fXOZProfilePos(0.),
                    fYOZProfilePos(0.),
                    fIsMoving(kFALSE),
                    fAntiAliasing(kTRUE),
                    fAxisPainter(axisPainter),
                    fTextureName(0),
                    fTexture(gDefTexture, gDefTexture + sizeof gDefTexture),
                    fBinWidth(1.)
{
   //If it's a standalone with gl context (not a pad),   
   //all initialization can be done one time here.

   if (MakeGLContextCurrent()) {
      gGLManager->ExtractViewport(fGLContext, fViewport);
      fArcBall.SetBounds(fViewport[2], fViewport[3]);
   }
}

//______________________________________________________________________________
void TGLLegoPainter::Paint()
{
   //
   if (!MakeGLContextCurrent())
      return;

   InitGL();
   //Save material/light properties in a stack.
   glPushAttrib(GL_LIGHTING_BIT);
   //I have to extract viewport each time,
   //because, for example, pad size can be changed
   //and Paint is called. If sizes were changed, 
   //selection buffer must be updated.
   const Int_t oldW = fViewport[2];
   const Int_t oldH = fViewport[3];
   gGLManager->ExtractViewport(fGLContext, fViewport);

   if (oldW != fViewport[2] || oldH != fViewport[3]) {
      fArcBall.SetBounds(fViewport[2], fViewport[3]);
      fUpdateSelection = kTRUE;
   }
   //glOrtho etc.
   SetCamera();
   //Clear buffer (possibly, with pad's background color).
   ClearBuffers();
   
   const Float_t pos[] = {0.f, 0.f, 0.f, 1.f};
   glLightfv(GL_LIGHT0, GL_POSITION, pos);
   //Set transformation - shift and rotate the scene.
   SetTransformation();
   FindFrontPoint();
   DrawPlot();
   //Restore material properties from stack.
   glPopAttrib();//(*)
   glFlush();
   //LegoPainter work is now finished, axes are drawn by axis painter.
   //Here changes are possible in future, if we have real 3d axis painter.
   gGLManager->ReadGLBuffer(fGLContext);
   if (fCoordType == kGLCartesian)
      fAxisPainter->Paint(fGLContext);
   gGLManager->Flush(fGLContext);
   /*
   //In future will be (starting from (*)):
      fAxisPainter->Paint(fGLContext);
      glFlush();
      gGLManager->Flush(fGLContext);
   */
}

//______________________________________________________________________________
void TGLLegoPainter::SetGLContext(Int_t ctx)
{
   //Called from TGLPadHistPainter (one painter can be in different pads).
   fGLContext = ctx;
}

//______________________________________________________________________________
char *TGLLegoPainter::GetObjectInfo(Int_t px, Int_t py)
{
   //This function is used by pad to show 
   //info in a status bar. It can be:
   //-bin number and content (selected bin is highlighted)
   //-TH name (back box planes are selected)
   //During rotation or shifting, this functions should
   //return immediately.
   if (fIsMoving)
      return "Shifting ...";
   //Convert from window top-bottom into gl bottom-top.
   py = fViewport[3] - py;
   //Y is a number of a row, x - column.
   std::swap(px, py);
   Selection_t newSelected(ColorToObject(fSelection.GetPixelColor(px, py)));
   fBinInfo = "";

   if (newSelected.first >= 0 && newSelected.second >= 0) {
      //There is a bin under cursor, show its info.
      fBinInfo.Form(
                    "(binx = %d; biny = %d; binc = %f)", 
                    newSelected.first + 1, 
                    newSelected.second + 1, 
                    fHist->GetBinContent(newSelected.first + 1, newSelected.second + 1)
                   );
      return (Char_t *)fBinInfo.Data();
   } else if (fSelectedPlane) {
      //Back plane or profile plane is selected.
      if (fHist->Class())
         fBinInfo += fHist->Class()->GetName();
      fBinInfo += "::";
      fBinInfo += fHist->GetName();

      return (Char_t *)fBinInfo.Data();
   }

   return " ";
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::InitGeometry()
{
   //Dispatch method.
   switch (fCoordType) {
   case kGLCartesian:
      return InitGeometryCartesian();
   case kGLPolar:
      return InitGeometryPolar();
   case kGLCylindrical:
      return InitGeometryCylindrical();
   case kGLSpherical:
      return InitGeometrySpherical();
   default:
      return kFALSE;
   }
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::InitGeometryCartesian()
{
   //Find bin ranges for X and Y axes,
   //axes ranges for X, Y and Z axes.
   //Function returns false, if log scale for
   //some axis was requested, but we cannot
   //find correct range.
   BinRange_t xBins, yBins;
   Range_t xRange, yRange, zRange;
   const TAxis *xAxis = fHist->GetXaxis();
   const TAxis *yAxis = fHist->GetYaxis();

   if (!ExtractAxisInfo(xAxis, fLogX, xBins, xRange)) {
      Error("TGLLegoPainter::InitGeometryCartesian", "Cannot set X axis to log scale");
      return kFALSE;
   }
   if (!ExtractAxisInfo(yAxis, fLogY, yBins, yRange)) {
      Error("TGLLegoPainter::InitGeometryCartesian", "Cannot set Y axis to log scale");
      return kFALSE;
   }
   if (!ExtractAxisZInfo(fHist, fLogZ, xBins, yBins, zRange))
   {
      Error("TGLLegoPainter::InitGeometryCartesian", 
            "Log scale is requested for Z, but maximum less or equal 0. (%f)", zRange.second);
      return kFALSE;
   }

   CalculateGLCameraParams(xRange, yRange, zRange);
   fAxisPainter->SetRanges(xRange, yRange, zRange);

   if (xBins != fBinsX || yBins != fBinsY || xRange != fRangeX || yRange != fRangeY || 
       zRange != fRangeZ || fBinWidth != fHist->GetBarWidth()) {
      fUpdateSelection = kTRUE;
      fXOZProfilePos = fFrame[0].Y();
      fYOZProfilePos = fFrame[0].X();
      fBinWidth = fHist->GetBarWidth();
   }

   fBinsX = xBins, fBinsY = yBins, fRangeX = xRange, fRangeY = yRange, fRangeZ = zRange;
   
   const Int_t nX = fBinsX.second - fBinsX.first + 1;
   fX.resize(nX + 1);

   if (fLogX)
      for (Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir)
         fX[i] = TMath::Log10(xAxis->GetBinLowEdge(ir)) * fScaleX;
   else
      for (Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir)
         fX[i] = xAxis->GetBinLowEdge(ir) * fScaleX;

   const Double_t maxX = xAxis->GetBinUpEdge(fBinsX.second);
   fLogX ? fX[nX] = TMath::Log10(maxX) * fScaleX : fX[nX] = maxX * fScaleX;

   const Int_t nY = fBinsY.second - fBinsY.first + 1;
   fY.resize(nY + 1);

   if (fLogY)
      for (Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr)
         fY[j] = TMath::Log10(yAxis->GetBinLowEdge(jr)) * fScaleY;
   else
      for (Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr)
         fY[j] = yAxis->GetBinLowEdge(jr) * fScaleY;

   const Double_t maxY = yAxis->GetBinUpEdge(fBinsY.second);
   fLogY ? fY[nY] = TMath::Log10(maxY) * fScaleY : fY[nY] = maxY * fScaleY;

   fMinZ = fFrame[0].Z();
   if (fMinZ < 0. && !fLogZ)
      fFrame[4].Z() > 0. ? fMinZ = 0. : fMinZ = fFrame[4].Z();

   fAxisPainter->SetZLevels(fZLevels);
   
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::InitGeometryPolar()
{
   //Find bin ranges for X and Y axes.
   //Find range for Z axis.
   //X is mapped to the polar angle,
   //Y to polar radius.
   //Z is Z.
   BinRange_t xBins, yBins;
   Range_t phiRange, roRange, zRange;
   const TAxis *xAxis = fHist->GetXaxis();
   const TAxis *yAxis = fHist->GetYaxis();

   if (!ExtractAxisInfo(xAxis, kFALSE, xBins, phiRange)) {
      Error("TGLLegoPainter::InitGeometryPolar", "Cannot set X axis to log scale");
      return kFALSE;
   }
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLLegoPainter::InitGeometryPolar", "To many PHI sectors");
      return kFALSE;
   }
   if (!ExtractAxisInfo(yAxis, kFALSE, yBins, roRange)) {
      Error("TGLLegoPainter::InitGeometryPolar", "Cannot set Y axis to log scale");
      return kFALSE;
   }
   if (!ExtractAxisZInfo(fHist, fLogZ, xBins, yBins, zRange))
   {
      Error("TGLLegoPainter::InitGeometryPolar", 
            "Log scale is requested for Z, but maximum less or equal 0. (%f)", zRange.second);
      return kFALSE;
   }

   CalculateGLCameraParams(Range_t(-1., 1.), Range_t(-1., 1.), zRange);

   if (xBins != fBinsX || yBins != fBinsY || phiRange != fRangeX || roRange != fRangeY || zRange != fRangeZ)
      fUpdateSelection = kTRUE;

   fBinsX = xBins, fBinsY = yBins, fRangeX = phiRange, fRangeY = roRange, fRangeZ = zRange;

   const Int_t nY = yBins.second - yBins.first + 1;
   fY.resize(nY + 1);
   const Double_t yLow = roRange.first;
   const Double_t maxRadius = roRange.second - roRange.first;

   for (Int_t j = 0, jr = yBins.first; j < nY; ++j, ++jr)
      fY[j] = ((yAxis->GetBinLowEdge(jr)) - yLow) / maxRadius * fScaleY;
   fY[nY] = (yAxis->GetBinUpEdge(fBinsY.second) - yLow) / maxRadius * fScaleY;

   const Int_t nX = xBins.second - xBins.first + 1;
   fCosSinTableX.resize(nX + 1);
   const Double_t fullAngle = xAxis->GetXmax() - xAxis->GetXmin();
   const Double_t phiLow = xAxis->GetXmin();
   Double_t angle = 0;
   for (Int_t i = 0, ir = xBins.first; i < nX; ++i, ++ir) {
      angle = (xAxis->GetBinLowEdge(ir) - phiLow) / fullAngle * TMath::TwoPi();
      fCosSinTableX[i].first = TMath::Cos(angle);
      fCosSinTableX[i].second = TMath::Sin(angle);
   }
   angle = (xAxis->GetBinUpEdge(fBinsX.second) - phiLow) / fullAngle * TMath::TwoPi();
   fCosSinTableX[nX].first = TMath::Cos(angle);
   fCosSinTableX[nX].second = TMath::Sin(angle);

   fMinZ = fFrame[0].Z();
   if (fMinZ < 0. && !fLogZ)
      fFrame[4].Z() > 0. ? fMinZ = 0. : fMinZ = fFrame[4].Z();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::InitGeometryCylindrical()
{
   //Find bin ranges for X and Y axes.
   //Find range for Z axis.
   //X is mapped to the azimuth,
   //Y is height.
   //Z is radius.
   BinRange_t xBins, yBins;
   Range_t angleRange, heightRange, radiusRange;
   const TAxis *xAxis = fHist->GetXaxis();
   const TAxis *yAxis = fHist->GetYaxis();

   ExtractAxisInfo(xAxis, kFALSE, xBins, angleRange);
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLLegoPainter::InitGeometryCylindrical", "To many PHI sectors");
      return kFALSE;
   }
   if (!ExtractAxisInfo(yAxis, fLogY, yBins, heightRange)) {
      Error("TGLLegoPainter::InitGeometryCylindrical", "Cannot set Y axis to log scale");
      return kFALSE;
   }
   ExtractAxisZInfo(fHist, kFALSE, xBins, yBins, radiusRange);

   CalculateGLCameraParams(Range_t(-1., 1.), Range_t(-1., 1.), heightRange);

   const Int_t nY = yBins.second - yBins.first + 1;
   fY.resize(nY + 1);

   if (fLogY)
      for (Int_t j = 0, jr = yBins.first; j < nY; ++j, ++jr)
         fY[j] = TMath::Log10(yAxis->GetBinLowEdge(jr)) * fScaleZ;
   else
      for (Int_t j = 0, jr = yBins.first; j < nY; ++j, ++jr)
         fY[j] = yAxis->GetBinLowEdge(jr) * fScaleZ;

   fLogY ? fY[nY] = TMath::Log10(yAxis->GetBinUpEdge(yBins.second)) * fScaleZ 
         : fY[nY] = yAxis->GetBinUpEdge(yBins.second) * fScaleZ;

   const Int_t nX = xBins.second - xBins.first + 1;
   fCosSinTableX.resize(nX + 1);
   const Double_t fullAngle = xAxis->GetXmax() - xAxis->GetXmin();
   const Double_t phiLow = xAxis->GetXmin();
   Double_t angle = 0.;
   for (Int_t i = 0, ir = xBins.first; i < nX; ++i, ++ir) {
      angle = (xAxis->GetBinLowEdge(ir) - phiLow) / fullAngle * TMath::TwoPi();
      fCosSinTableX[i].first = TMath::Cos(angle);
      fCosSinTableX[i].second = TMath::Sin(angle);
   }
   angle = (xAxis->GetBinUpEdge(fBinsX.second) - phiLow) / fullAngle * TMath::TwoPi();
   fCosSinTableX[nX].first = TMath::Cos(angle);
   fCosSinTableX[nX].second = TMath::Sin(angle);

   if (xBins != fBinsX || yBins != fBinsY || angleRange != fRangeX || heightRange != fRangeY || radiusRange != fRangeZ)
      fUpdateSelection = kTRUE;

   fBinsX = xBins, fBinsY = yBins, fRangeX = angleRange, fRangeY = heightRange, fRangeZ = radiusRange;

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::InitGeometrySpherical()
{
   //Find bin ranges for X and Y axes.
   //Find range for Z axis.
   //X is mapped to the theta,
   //Y is phi,
   //Z is radius.
   BinRange_t xBins, yBins;
   Range_t phiRange, thetaRange, radiusRange;
   const TAxis *xAxis = fHist->GetXaxis();
   const TAxis *yAxis = fHist->GetYaxis();

   ExtractAxisInfo(xAxis, kFALSE, xBins, phiRange);
   if (xBins.second - xBins.first + 1 > 360) {
      Error("TGLLegoPainter::InitGeometrySpherical", "To many PHI sectors");
      return kFALSE;
   }
   ExtractAxisInfo(yAxis, kFALSE, yBins, thetaRange);
   if (yBins.second - yBins.first + 1 > 180) {
      Error("TGLLegoPainter::InitGeometrySpherical", "To many THETA sectors");
      return kFALSE;
   }
   ExtractAxisZInfo(fHist, kFALSE, xBins, yBins, radiusRange);

   CalculateGLCameraParams(Range_t(-1., 1.), Range_t(-1., 1.), Range_t(-1., 1.));

   const Int_t nY = yBins.second - yBins.first + 1;
   fCosSinTableY.resize(nY + 1);
   const Double_t fullTheta = yAxis->GetXmax() - yAxis->GetXmin();
   const Double_t thetaLow = yAxis->GetXmin();
   Double_t angle = 0.;
   for (Int_t j = 0, jr = yBins.first; j < nY; ++j, ++jr) {
      angle = (yAxis->GetBinLowEdge(jr) - thetaLow) / fullTheta * TMath::Pi();
      fCosSinTableY[j].first = TMath::Cos(angle);
      fCosSinTableY[j].second = TMath::Sin(angle);
   }
   angle = (yAxis->GetBinUpEdge(fBinsY.second) - thetaLow) / fullTheta * TMath::Pi();
   fCosSinTableY[nY].first = TMath::Cos(angle);
   fCosSinTableY[nY].second = TMath::Sin(angle);

   const Int_t nX = xBins.second - xBins.first + 1;
   fCosSinTableX.resize(nX + 1);
   const Double_t fullPhi = xAxis->GetXmax() - xAxis->GetXmin();
   const Double_t phiLow = xAxis->GetXmin();

   for (Int_t i = 0, ir = xBins.first; i < nX; ++i, ++ir) {
      angle = (xAxis->GetBinLowEdge(ir) - phiLow) / fullPhi * TMath::TwoPi();
      fCosSinTableX[i].first = TMath::Cos(angle);
      fCosSinTableX[i].second = TMath::Sin(angle);
   }
   angle = (xAxis->GetBinUpEdge(fBinsX.second) - phiLow) / fullPhi * TMath::TwoPi();
   fCosSinTableX[nX].first = TMath::Cos(angle);
   fCosSinTableX[nX].second = TMath::Sin(angle);

   if (xBins != fBinsX || yBins != fBinsY || phiRange != fRangeX || thetaRange != fRangeY || radiusRange != fRangeZ)
      fUpdateSelection = kTRUE;

   fBinsX = xBins, fBinsY = yBins, fRangeX = phiRange, fRangeY = thetaRange, fRangeZ = radiusRange;

   return kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::StartRotation(Int_t px, Int_t py)
{
   //Rotation started.
   fArcBall.Click(TPoint(px, py));
   //GetObjectInfo must be disabled (in a pad).
   fIsMoving = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::Rotate(Int_t px, Int_t py)
{
   //Rotation.
   //Selection buffer must be updated.
   fArcBall.Drag(TPoint(px, py));
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::StopRotation()
{
   //Rotation is finished.
   fIsMoving = kFALSE;
}

//______________________________________________________________________________
void TGLLegoPainter::StartPan(Int_t px, Int_t py)
{
   //User clicks right mouse button (in a pad).
   fMousePosition.fX = px;
   fMousePosition.fY = fViewport[3] - py;
   //Disable GetObjectInfo.
   fIsMoving = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::Pan(Int_t px, Int_t py)
{
   //User's moving mouse cursor, with right mouse button pressed (for pad).
   //Calculate 3d shift related to 2d mouse movement.
   if (!MakeGLContextCurrent())
      return;

   //Convert py into bottom-top orientation.
   py = fViewport[3] - py;

   if (fSelectedPlane < 2)//Shift lego.
      AdjustShift(fMousePosition, TPoint(px, py), fPan, fViewport);
   else if (fSelectedPlane >= 2 && fSelectedPlane <= 5)
      MoveDynamicProfile(px, py);

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::StopPan()
{
   //Right mouse button was released (in a pad).
   //Enable GetObjectInfo.
   fIsMoving = kFALSE;
}

//______________________________________________________________________________
TObject *TGLLegoPainter::Select(Int_t px, Int_t py)
{
   //
   if (!MakeGLContextCurrent())
      return 0;
   //Read color buffer content, to find selected object
   if (fUpdateSelection) {
      SetSelectionMode();
      fSelectionPass = kTRUE;
      SetCamera();
      glDisable(GL_LIGHTING);
      glClearColor(0.f, 0.f, 0.f, 0.f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
      SetTransformation();
      DrawPlot();
      glFlush();
      fSelection.ReadColorBuffer(fViewport[2], fViewport[3]);
      fSelectionPass = kFALSE;
      fUpdateSelection = kFALSE;
      glEnable(GL_LIGHTING);
   }
   //Convert from window top-bottom into gl bottom-top.
   py = fViewport[3] - py;
   //Y is a number of a row, x - column.
   std::swap(px, py);
   Selection_t newSelected(ColorToObject(fSelection.GetPixelColor(px, py)));

   if (newSelected != fSelectedBin) {
      //New object was selected (or lego deselected) - re-paint.
      fSelectedBin = newSelected;
      gGLManager->MarkForDirectCopy(fGLContext, kTRUE);
      Paint();
      gGLManager->MarkForDirectCopy(fGLContext, kFALSE);
   }
   //If cursor is in the lego's area, TH object will be selected.
   return fSelectedBin.first != -1 || fSelectedPlane ? fHist : 0;
}

//______________________________________________________________________________
void TGLLegoPainter::ZoomIn()
{
   //+Zoom
   fZoom /= 1.2;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::ZoomOut()
{
   //-Zoom
   fZoom *= 1.2;
   fUpdateSelection = kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::SetLogX(Bool_t log)
{
   //X logarithmic scale.
   fCoordType == kGLCartesian ? fLogX = log : fLogX = kFALSE;
}

//______________________________________________________________________________
void TGLLegoPainter::SetLogY(Bool_t log)
{
   //Y logarithmic scale.
   fCoordType == kGLCartesian || fCoordType == kGLCylindrical ? fLogY = log : fLogY = kFALSE;
}

//______________________________________________________________________________
void TGLLegoPainter::SetLogZ(Bool_t log)
{
   //Z logarithmic scale.
   fCoordType == kGLCartesian || fCoordType == kGLPolar ? fLogZ = log : fLogZ = kFALSE;
}

//______________________________________________________________________________
void TGLLegoPainter::SetCoordType(EGLCoordType type)
{
   //kGLCartesian or kGLPolar etc.
   if (type != fCoordType)
      fUpdateSelection = kTRUE;
   fCoordType = type;
}

//______________________________________________________________________________
void TGLLegoPainter::AddOption(const TString &option)
{
   //Set lego type from the string option.
   const Ssiz_t legoPos = option.Index("lego");
   if (legoPos + 4 < option.Length() && std::isdigit(option[legoPos + 4])) {
      switch (option[legoPos + 4] - '0') {
      case 1:
         fLegoType = kColorSimple;
         break;
      case 2:
         fLegoType = kColorLevel;
         break;
      case 3:
         fLegoType = kCylindricBars;
         break;
      }
   } else
      fLegoType = kColorSimple;
   //check 'e' option 
}

//______________________________________________________________________________
void TGLLegoPainter::SetPadColor(TColor *c)
{
   //Used only in a pad.
   fPadColor = c;
}

//______________________________________________________________________________
void TGLLegoPainter::SetFrameColor(TColor *c)
{
   //Used only in a pad.
   fFrameColor = c;
}


//______________________________________________________________________________
void TGLLegoPainter::InitGL()
{
   //Common gl initialization.
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   //For lego, back polygons are culled (but not for dynamic profiles).
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

//______________________________________________________________________________
TGLLegoPainter::Selection_t TGLLegoPainter::ColorToObject(const UChar_t *color)
{
   //"Object" is a bin number ([xBin, yBin]),
   //plane (2 - left, 3 - right, 1 - bottom, 4 - XOZ dynamic profile, 
   //5 - YOZ dynamic profile). Function returns bin number pair or
   //sets fPlaneSelected (in this case, number pair will be [-1, -1] - 
   //no bin was selected). 
   const Int_t nY = fBinsY.second - fBinsY.first + 1;
   const Int_t nBins = nY * (fBinsX.second - fBinsX.first + 1);
   Selection_t selected(-1, -1);

   if (fSelectionMode == kSelectionSimple) {
      //Simple mode - hist can be selected only
      //as whole, bins/planes/profiles can not be selected.
      if (color[0])
         fSelectedPlane = 1;
   } else {
      if (Int_t selectedNum = color[0] | (color[1] << 8) | (color[2] << 16)) {
         //Remove additional 1 (look at EncodeToColor definition).
         selectedNum -= 1;
         if (selectedNum < nBins) {//Bin was selected.
            fSelectedPlane = 0;    //deselect plane (if it was selected)
            selected.first = selectedNum / nY;
            selected.second = selectedNum % nY;
         } else {
            selectedNum -= nBins;
            if (fSelectedPlane != selectedNum) {
               fSelectedPlane = selectedNum;
               //Selection changed, re-paint.
               gGLManager->MarkForDirectCopy(fGLContext, kTRUE);
               Paint();
               gGLManager->MarkForDirectCopy(fGLContext, kFALSE);
            }
         }
      } else if (fSelectedPlane) {
         //Selection changed, re-paint.
         fSelectedPlane = 0;
         gGLManager->MarkForDirectCopy(fGLContext, kTRUE);
         Paint();
         gGLManager->MarkForDirectCopy(fGLContext, kFALSE);
      }
   }

   return selected;
}

//______________________________________________________________________________
void TGLLegoPainter::EncodeToColor(Int_t i, Int_t j)const
{
   //In a simple mode, all lego (with it's frame) is white (in a black color buffer)
   if (fSelectionMode == kSelectionSimple)
      glColor3ub(255, 255, 255);
   else {
      //Two bin indicies are converted into a number, this number is encoded
      //as r, g, b values. + 1  after j => for bin [0, 0] 
      //(to distinguish from black color buffer).
      const Int_t code = i * (fBinsY.second - fBinsY.first + 1) + j + 1;
      glColor3ub(code & 0xff, (code & 0xff00) >> 8, (code & 0xff0000) >> 16);
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawPlot()
{
   //Select actual drawing function.
   switch (fCoordType) {
   case kGLPolar:
      return DrawLegoPolar();
   case kGLCylindrical:
      return DrawLegoCylindrical();
   case kGLSpherical:
      return DrawLegoSpherical();
   case kGLCartesian:
   default:
      DrawLegoCartesian();
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawLegoCartesian()
{
   //Draws lego in a cartesian system.
   const Int_t nX = fX.size() - 1;
   const Int_t nY = fY.size() - 1;
   //Draw back box (possibly, with dynamic profiles).
   DrawFrame();

   if (!fSelectionPass) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[0
      glPolygonOffset(1.f, 1.f);
      SetLegoColor();
      if (fXOZProfilePos > fFrame[0].Y() || fYOZProfilePos > fFrame[0].X()) {
         //Lego is semi-transparent during dynamic profiling
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      }
   }

   //Using front point, find the correct order to draw bars from
   //back to front (it's important only for semi-transparent lego).
   //Only in cartesian.
   Int_t iInit = 0, jInit = 0, irInit = fBinsX.first, jrInit = fBinsY.first;
   const Int_t addI = fFrontPoint == 2 || fFrontPoint == 1 ? 1 : (iInit = nX - 1, irInit = fBinsX.second, -1);
   const Int_t addJ = fFrontPoint == 2 || fFrontPoint == 3 ? 1 : (jInit = nY - 1, jrInit = fBinsY.second, -1);

   if (fLegoType == kColorLevel && !fSelectionPass)
      Enable1DTexture();

   for(Int_t i = iInit, ir = irInit; addI > 0 ? i < nX : i >= 0; i += addI, ir += addI) {
      for(Int_t j = jInit, jr = jrInit; addJ > 0 ? j < nY : j >= 0; j += addJ, jr += addJ) {
         Double_t zMax = fHist->GetCellContent(ir, jr) * fFactor;
         if (!ClampZ(zMax))
            continue;
         if (fSelectionPass)
            EncodeToColor(i, j);
         else if(fSelectedBin == Selection_t(i, j) && fSelectionMode == kSelectionFull)
            glMaterialfv(GL_FRONT, GL_EMISSION, fRedEmission);
         
         Double_t xMin = fX[i], xMax = fX[i + 1], yMin = fY[j], yMax = fY[j + 1];//

         if (fBinWidth < 1.) {
            Double_t xW = xMax - xMin;
            xMin = xMin + xW / 2 - xW * fBinWidth / 2, xMax = xMin + xW * fBinWidth;
            Double_t yW = yMax - yMin;
            yMin = yMin + yW / 2 - yW * fBinWidth / 2, yMax = yMin + yW * fBinWidth;
         }

         if (fLegoType == kCylindricBars)
            //RootGL::DrawCylinder(&fQuadric, fX[i], fX[i + 1], fY[j], fY[j + 1], fMinZ, zMax);
            RootGL::DrawCylinder(&fQuadric, xMin, xMax, yMin, yMax, fMinZ, zMax);
         else if (fLegoType == kColorLevel && !fSelectionPass) {
            const Double_t zRange = fRangeZ.second - fRangeZ.first;
            RootGL::DrawBoxFrontTextured(
                                         //fX[i], fX[i + 1], fY[j], fY[j + 1], fMinZ, zMax, 
                                         xMin, xMax, yMin, yMax, fMinZ, zMax,
                                         (fMinZ - fRangeZ.first) / zRange, 
                                         (zMax  - fRangeZ.first) / zRange, 
                                         fFrontPoint
                                        );
         }
         else
            //RootGL::DrawBoxFront(fX[i], fX[i + 1], fY[j], fY[j + 1], fMinZ, zMax, fFrontPoint);
            RootGL::DrawBoxFront(xMin, xMax, yMin, yMax, fMinZ, zMax, fFrontPoint);
     
         if(fSelectedBin == Selection_t(i, j) && fSelectionMode == kSelectionFull)
            glMaterialfv(GL_FRONT, GL_EMISSION, fNullEmission);
      }
   }

   if (fLegoType == kColorLevel && !fSelectionPass)
      Disable1DTexture();

   //Draw outlines for non-cylindrical bars.         
   if (!fSelectionPass && fLegoType != kCylindricBars) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      RootGL::TGLDisableGuard lightGuard(GL_LIGHTING);//[2 - 2]
      if (fXOZProfilePos <= fFrame[0].Y() && fYOZProfilePos <= fFrame[0].X())
         glColor3d(0., 0., 0.);
      else
         glColor4d(0., 0., 0., 0.4);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      if (fAntiAliasing) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glEnable(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      }

      for(Int_t i = iInit, ir = irInit; addI > 0 ? i < nX : i >= 0; i += addI, ir += addI) {
         for(Int_t j = jInit, jr = jrInit; addJ > 0 ? j < nY : j >= 0; j += addJ, jr += addJ) {
            Double_t zMax = fHist->GetCellContent(ir, jr) * fFactor;
            if (!ClampZ(zMax))
               continue;

            Double_t xMin = fX[i], xMax = fX[i + 1], yMin = fY[j], yMax = fY[j + 1];//

            if (fBinWidth < 1.) {
               Double_t xW = xMax - xMin;
               xMin = xMin + xW / 2 - xW * fBinWidth / 2, xMax = xMin + xW * fBinWidth;
               Double_t yW = yMax - yMin;
               yMin = yMin + yW / 2 - yW * fBinWidth / 2, yMax = yMin + yW * fBinWidth;
            }

            //RootGL::DrawBoxFront(fX[i], fX[i + 1], fY[j], fY[j + 1], fMinZ, zMax, fFrontPoint);
            RootGL::DrawBoxFront(xMin, xMax, yMin, yMax, fMinZ, zMax, fFrontPoint);
         }
      }

      if (fAntiAliasing) {
         glDisable(GL_BLEND);
         glDisable(GL_LINE_SMOOTH);
      }

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawLegoPolar()
{
   //Draws lego in a polar system.
   //No back box, no profiles.
   //Bars are drawn as trapezoids.
   const Int_t nX = fCosSinTableX.size() - 1;
   const Int_t nY = fY.size() - 1;

   if (!fSelectionPass) {
      SetLegoColor();
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
   }

   Double_t points[4][2] = {};

   if (!fSelectionPass && fLegoType == kColorLevel)
      Enable1DTexture();

   for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
      for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
         Double_t zMax = fHist->GetCellContent(ir, jr) * fFactor;
         if (!ClampZ(zMax))
            continue;
         points[0][0] = fY[j] * fCosSinTableX[i].first;
         points[0][1] = fY[j] * fCosSinTableX[i].second;
         points[1][0] = fY[j + 1] * fCosSinTableX[i].first;
         points[1][1] = fY[j + 1] * fCosSinTableX[i].second;
         points[2][0] = fY[j + 1] * fCosSinTableX[i + 1].first;
         points[2][1] = fY[j + 1] * fCosSinTableX[i + 1].second;
         points[3][0] = fY[j] * fCosSinTableX[i + 1].first;
         points[3][1] = fY[j] * fCosSinTableX[i + 1].second;
         if (fSelectionPass)
            EncodeToColor(i, j);
         else if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fRedEmission);

         if (fLegoType == kColorLevel && !fSelectionPass) {
            const Double_t zRange = fRangeZ.second - fRangeZ.first;
            RootGL::DrawTrapezoidTextured(
                                          points, fMinZ, zMax, 
                                          (fMinZ - fRangeZ.first) / zRange, 
                                          (zMax  - fRangeZ.first) / zRange
                                         );

         }
         else
            RootGL::DrawTrapezoid(points, fMinZ, zMax);

         if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fNullEmission);
      }
   }

   if (fLegoType == kColorLevel && !fSelectionPass)
      Disable1DTexture();

   //Draw otulines.
   if (!fSelectionPass) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      RootGL::TGLDisableGuard lightGuard(GL_LIGHTING);//[2-2]
      glColor3d(0., 0., 0.);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      if (fAntiAliasing) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glEnable(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      }

      for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
         for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
            Double_t zMax = fHist->GetCellContent(ir, jr) * fFactor;
            if (!ClampZ(zMax))
               continue;
            points[0][0] = fY[j] * fCosSinTableX[i].first;
            points[0][1] = fY[j] * fCosSinTableX[i].second;
            points[1][0] = fY[j + 1] * fCosSinTableX[i].first;
            points[1][1] = fY[j + 1] * fCosSinTableX[i].second;
            points[2][0] = fY[j + 1] * fCosSinTableX[i + 1].first;
            points[2][1] = fY[j + 1] * fCosSinTableX[i + 1].second;
            points[3][0] = fY[j] * fCosSinTableX[i + 1].first;
            points[3][1] = fY[j] * fCosSinTableX[i + 1].second;
            RootGL::DrawTrapezoid(points, fMinZ, zMax, kFALSE);
         }
      }

      if (fAntiAliasing) {
         glDisable(GL_BLEND);
         glDisable(GL_LINE_SMOOTH);
      }

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawLegoCylindrical()
{
   //
   const Int_t nX = fCosSinTableX.size() - 1;
   const Int_t nY = fY.size() - 1;
   const Double_t rRange = fRangeZ.second - fRangeZ.first;
   Double_t legoR = gStyle->GetLegoInnerR();
   if (legoR > 1. || legoR < 0.)
      legoR = 0.5;

   if (!fSelectionPass) {
      SetLegoColor();
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
   }

   Double_t points[4][2] = {};
   const Double_t sc = (1 - legoR) * fScaleX;
   Double_t zMax = 0, zMin = 0, zVal = 0;

   if (!fSelectionPass && fLegoType == kColorLevel)
      Enable1DTexture();

   for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
      for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
         zVal = fHist->GetCellContent(ir, jr) * fFactor;
         if (zVal >= 0.) {
            zMin = legoR * fScaleX;
            zMax = legoR * fScaleX + zVal / rRange * sc;
         } else {
            zMax = legoR * fScaleX;
            zMin = legoR * fScaleX + zVal / rRange * sc;
         }

         points[0][0] = fCosSinTableX[i].first * zMin;
         points[0][1] = fCosSinTableX[i].second * zMin;
         points[1][0] = fCosSinTableX[i].first * zMax;
         points[1][1] = fCosSinTableX[i].second * zMax;
         points[2][0] = fCosSinTableX[i + 1].first * zMax;
         points[2][1] = fCosSinTableX[i + 1].second * zMax;
         points[3][0] = fCosSinTableX[i + 1].first * zMin;
         points[3][1] = fCosSinTableX[i + 1].second * zMin;

         if (fSelectionPass)
            EncodeToColor(i, j);
         else if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fRedEmission);

         if (fLegoType == kColorLevel && !fSelectionPass) {
            const Double_t zRange = fRangeZ.second - fRangeZ.first;
            RootGL::DrawTrapezoidTextured2(
                                           points, fY[j], fY[j + 1], 
                                           (fMinZ - fRangeZ.first) / zRange, 
                                           (zVal  - fRangeZ.first) / zRange
                                          );
         }
         else
            RootGL::DrawTrapezoid(points, fY[j], fY[j + 1]);

         if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fNullEmission);
      }
   }

   if (!fSelectionPass && fLegoType == kColorLevel)
      Disable1DTexture();

   //Draw otulines.
   if (!fSelectionPass) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      RootGL::TGLDisableGuard lightGuard(GL_LIGHTING);//[2-2]
      glColor3d(0., 0., 0.);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      if (fAntiAliasing) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glEnable(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      }

      for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
         for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
            zVal = fHist->GetCellContent(ir, jr) * fFactor;
            if (zVal >= 0.) {
               zMin = legoR * fScaleX;
               zMax = legoR * fScaleX + zVal / rRange * sc;
            } else {
               zMax = legoR * fScaleX;
               zMin = legoR * fScaleX + zVal / rRange * sc;
            }

            points[0][0] = fCosSinTableX[i].first * zMin;
            points[0][1] = fCosSinTableX[i].second * zMin;
            points[1][0] = fCosSinTableX[i].first * zMax;
            points[1][1] = fCosSinTableX[i].second * zMax;
            points[2][0] = fCosSinTableX[i + 1].first * zMax;
            points[2][1] = fCosSinTableX[i + 1].second * zMax;
            points[3][0] = fCosSinTableX[i + 1].first * zMin;
            points[3][1] = fCosSinTableX[i + 1].second * zMin;
            RootGL::DrawTrapezoid(points, fY[j], fY[j + 1]);
         }
      }

      if (fAntiAliasing) {
         glDisable(GL_BLEND);
         glDisable(GL_LINE_SMOOTH);
      }

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawLegoSpherical()
{
   //
   const Int_t nX = fCosSinTableX.size() - 1;
   const Int_t nY = fCosSinTableY.size() - 1;
   const Double_t rRange = fRangeZ.second - fRangeZ.first;
   Double_t legoR = gStyle->GetLegoInnerR();
   if (legoR > 1. || legoR < 0.)
      legoR = 0.5;

   if (!fSelectionPass) {
      SetLegoColor();
      glEnable(GL_POLYGON_OFFSET_FILL);
      glPolygonOffset(1.f, 1.f);
   }

   Double_t points[8][3] = {};
   const Double_t sc = (1 - legoR) * fScaleX;
   Double_t zMax = 0, zMin = 0, zVal = 0;

   if (!fSelectionPass && fLegoType == kColorLevel)
      Enable1DTexture();

   for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
      for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
         zVal = fHist->GetCellContent(ir, jr) * fFactor;
         if (zVal >= 0.) {
            zMin = legoR * fScaleX;
            zMax = legoR * fScaleX + zVal / rRange * sc;
         } else {
            zMax = legoR * fScaleX;
            zMin = legoR * fScaleX + zVal / rRange * sc;
         }

         points[4][0] = zMin * fCosSinTableY[j].second * fCosSinTableX[i].first;
         points[4][1] = zMin * fCosSinTableY[j].second * fCosSinTableX[i].second;
         points[4][2] = zMin * fCosSinTableY[j].first;
         points[5][0] = zMin * fCosSinTableY[j].second * fCosSinTableX[i + 1].first;
         points[5][1] = zMin * fCosSinTableY[j].second * fCosSinTableX[i + 1].second;
         points[5][2] = zMin * fCosSinTableY[j].first;
         points[6][0] = zMax * fCosSinTableY[j].second * fCosSinTableX[i + 1].first;
         points[6][1] = zMax * fCosSinTableY[j].second * fCosSinTableX[i + 1].second;
         points[6][2] = zMax * fCosSinTableY[j].first;
         points[7][0] = zMax * fCosSinTableY[j].second * fCosSinTableX[i].first;
         points[7][1] = zMax * fCosSinTableY[j].second * fCosSinTableX[i].second;
         points[7][2] = zMax * fCosSinTableY[j].first;
         points[0][0] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i].first;
         points[0][1] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i].second;
         points[0][2] = zMin * fCosSinTableY[j + 1].first;
         points[1][0] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].first;
         points[1][1] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].second;
         points[1][2] = zMin * fCosSinTableY[j + 1].first;
         points[2][0] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].first;
         points[2][1] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].second;
         points[2][2] = zMax * fCosSinTableY[j + 1].first;
         points[3][0] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i].first;
         points[3][1] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i].second;
         points[3][2] = zMax * fCosSinTableY[j + 1].first;

         
         if (fSelectionPass)
            EncodeToColor(i, j);
         else if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fRedEmission);
         if (fLegoType == kColorLevel && !fSelectionPass) {
            const Double_t zRange = fRangeZ.second - fRangeZ.first;
            RootGL::DrawTrapezoidTextured(
                                          points, 
                                          (fMinZ - fRangeZ.first) / zRange, 
                                          (zVal  - fRangeZ.first) / zRange
                                         );
         }
         else
            RootGL::DrawTrapezoid(points);

         if(fSelectedBin == Selection_t(i, j))
            glMaterialfv(GL_FRONT, GL_EMISSION, fNullEmission);
      }
   }

   if (!fSelectionPass && fLegoType == kColorLevel)
      Disable1DTexture();

   //Draw otulines.
   if (!fSelectionPass) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      RootGL::TGLDisableGuard lightGuard(GL_LIGHTING);//[2-2]
      glColor3d(0., 0., 0.);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      if (fAntiAliasing) {
         glEnable(GL_BLEND);
         glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
         glEnable(GL_LINE_SMOOTH);
         glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      }

      for(Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
         for(Int_t j = 0, jr = fBinsY.first; j < nY; ++j, ++jr) {
            zVal = fHist->GetCellContent(ir, jr) * fFactor;
            if (zVal >= 0.) {
               zMin = legoR * fScaleX;
               zMax = legoR * fScaleX + zVal / rRange * sc;
            } else {
               zMax = legoR * fScaleX;
               zMin = legoR * fScaleX + zVal / rRange * sc;
            }

            points[4][0] = zMin * fCosSinTableY[j].second * fCosSinTableX[i].first;
            points[4][1] = zMin * fCosSinTableY[j].second * fCosSinTableX[i].second;
            points[4][2] = zMin * fCosSinTableY[j].first;
            points[5][0] = zMin * fCosSinTableY[j].second * fCosSinTableX[i + 1].first;
            points[5][1] = zMin * fCosSinTableY[j].second * fCosSinTableX[i + 1].second;
            points[5][2] = zMin * fCosSinTableY[j].first;
            points[6][0] = zMax * fCosSinTableY[j].second * fCosSinTableX[i + 1].first;
            points[6][1] = zMax * fCosSinTableY[j].second * fCosSinTableX[i + 1].second;
            points[6][2] = zMax * fCosSinTableY[j].first;
            points[7][0] = zMax * fCosSinTableY[j].second * fCosSinTableX[i].first;
            points[7][1] = zMax * fCosSinTableY[j].second * fCosSinTableX[i].second;
            points[7][2] = zMax * fCosSinTableY[j].first;
            points[0][0] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i].first;
            points[0][1] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i].second;
            points[0][2] = zMin * fCosSinTableY[j + 1].first;
            points[1][0] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].first;
            points[1][1] = zMin * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].second;
            points[1][2] = zMin * fCosSinTableY[j + 1].first;
            points[2][0] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].first;
            points[2][1] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i + 1].second;
            points[2][2] = zMax * fCosSinTableY[j + 1].first;
            points[3][0] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i].first;
            points[3][1] = zMax * fCosSinTableY[j + 1].second * fCosSinTableX[i].second;
            points[3][2] = zMax * fCosSinTableY[j + 1].first;
            RootGL::DrawTrapezoid(points);
         }
      }

      if (fAntiAliasing) {
         glDisable(GL_BLEND);
         glDisable(GL_LINE_SMOOTH);
      }

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }
}

//______________________________________________________________________________
void TGLLegoPainter::SetLegoColor()
{
   //Set color for lego.
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.15f};

   if (fLegoType != kColorLevel && fHist->GetFillColor() != kWhite)
      if (TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   
   glMaterialfv(GL_FRONT, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT, GL_SHININESS, 70.f);
}

Bool_t TGLLegoPainter::MakeGLContextCurrent()const
{
   //Make the gl context current.
   return fGLContext != -1 && gGLManager->MakeCurrent(fGLContext);
}

//______________________________________________________________________________
void TGLLegoPainter::ClearBuffers()
{
   //Clears gl buffers (possibly with pad's background color).
   Float_t rgb[3] = {1.f, 1.f, 1.f};
   if (fPadColor)
      fPadColor->GetRGB(rgb[0], rgb[1], rgb[2]);
   glClearColor(rgb[0], rgb[1], rgb[2], 1.);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

//______________________________________________________________________________
void TGLLegoPainter::SetSelectionMode()
{
   //Number of bins + 5 must be less then 2^24 (5 == 3 back planes + 2 dynamic profiles).
   //2 ^ 24 == r g b in a glColor3ub (ub for unsigned char). Number of bits supposed 
   //== 8.
   if ((fBinsX.second - fBinsX.first) * (fBinsY.second - fBinsY.first) > (1u<<24) - 5)
      fSelectionMode = kSelectionSimple;
   else
      fSelectionMode = kSelectionFull;
}

//______________________________________________________________________________
void TGLLegoPainter::DrawFrame()
{
   //Draw back box for lego.
   //This box is not written into
   //OpenGL's depth buffer, to avoid some visual artifacts.
   if (!fSelectionPass) {
      glEnable(GL_BLEND);//[0
      glDepthMask(GL_FALSE);//[1
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }

   //Back plane is partially transparent to make its color smoother.
   Float_t backColor[] = {0.9f, 0.9f, 0.9f, 0.85f};
   if (fFrameColor)
      fFrameColor->GetRGB(backColor[0], backColor[1], backColor[2]);
   //Planes are encoded as number of bins + plane number (1,2,3)
   const Int_t selectionBase = fBinsX.second - fBinsX.first + 1;

   if (!fSelectionPass)
      glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);
   else
      EncodeToColor(selectionBase, 1);//bottom plane

   RootGL::DrawQuadFilled(fFrame[0], fFrame[1], fFrame[2], fFrame[3], TGLVertex3(0., 0., 1.));

   //Left plane, encoded as 2 + number of bins in a selection buffer.
   if (!fSelectionPass) {
      if (fSelectedPlane == 2 && fSelectionMode == kSelectionFull)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, fGreenEmission);
   } else
      EncodeToColor(selectionBase, 2);
   DrawBackPlane(fBackPairs[fFrontPoint][0]);
   if (!fSelectionPass) {
      if (fSelectedPlane == 2 && fSelectionMode == kSelectionFull)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, fNullEmission);
   }

   //Right plane, encoded as 3 in a selection buffer.
   if (!fSelectionPass) {
      glMaterialfv(GL_FRONT, GL_DIFFUSE, backColor);
      if (fSelectedPlane == 3 && fSelectionMode == kSelectionFull)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, fGreenEmission);
   } else
      EncodeToColor(selectionBase, 3);
   DrawBackPlane(fBackPairs[fFrontPoint][1]);

   if (!fSelectionPass) {
      if (fSelectedPlane == 3 && fSelectionMode == kSelectionFull)
         glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, fNullEmission);
      glDepthMask(GL_TRUE);//1]
      glDisable(GL_BLEND);//0]
   }

   DrawProfiles();
}

//______________________________________________________________________________
void TGLLegoPainter::DrawBackPlane(Int_t plane)const
{
   //Draw back plane with number 'plane'
   const Int_t *vertInd = fFramePlanes[plane];
   TGLVertex3 normal(fFrameNormals[plane][0], fFrameNormals[plane][1], fFrameNormals[plane][2]);
   RootGL::DrawQuadFilled(fFrame[vertInd[0]], fFrame[vertInd[1]], fFrame[vertInd[2]], fFrame[vertInd[3]], normal);
   //antialias back plane outline
   if (!fSelectionPass) {
      using namespace RootGL;
      TGLEnableGuard lineGuard(GL_LINE_SMOOTH);
      TGLEnableGuard blendGuard(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      TGLDisableGuard depthGuard(GL_DEPTH_TEST);
      TGLDisableGuard lightGuard(GL_LIGHTING);
      glColor3d(0., 0., 0.);
      DrawQuadOutline(fFrame[vertInd[0]], fFrame[vertInd[1]], fFrame[vertInd[2]], fFrame[vertInd[3]]);
      DrawGrid(plane);
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawProfiles()
{
   //Draw static profiles ("shadows") and
   //dynamic profile (if any).
   using namespace RootGL;
   const Int_t selectionBase = fBinsX.second - fBinsX.first + 1;

   if (fXOZProfilePos > fFrame[0].Y()) {
      if (fXOZProfilePos > fFrame[2].Y())
         fXOZProfilePos = fFrame[2].Y();
      TGLDisableGuard cullGuard(GL_CULL_FACE);
      TGLVertex3 v1(fFrame[0].X(), fXOZProfilePos, fFrame[0].Z());
      TGLVertex3 v2(fFrame[1].X(), fXOZProfilePos, fFrame[1].Z());
      TGLVertex3 v3(fFrame[5].X(), fXOZProfilePos, fFrame[5].Z());
      TGLVertex3 v4(fFrame[4].X(), fXOZProfilePos, fFrame[4].Z());

      if (fSelectionPass)
         EncodeToColor(selectionBase, 4);
      else {
         glDisable(GL_LIGHTING);

         if (fSelectedPlane == 4) {
            TGLEnableGuard blendGuard(GL_BLEND);
            TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(3.f);
            glColor3d(0., 0.6, 1.);
            DrawQuadOutline(v1, v2, v3, v4);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glLineWidth(1.f);
         }
         glColor3d(0.6, 0.6, 0.6);
      }
      RootGL::DrawQuadFilled(v1, v2, v3, v4, TGLVertex3(0., 1., 0.));
      TGLDisableGuard depth(GL_DEPTH_TEST);
      DrawProfileX();
      if (!fSelectionPass)
         glEnable(GL_LIGHTING);
   }

   if (fYOZProfilePos > fFrame[0].X()) {
      if (fYOZProfilePos > fFrame[1].X())
         fYOZProfilePos = fFrame[1].X();
      TGLDisableGuard cullGuard(GL_CULL_FACE);
      TGLVertex3 v1(fYOZProfilePos, fFrame[0].Y(), fFrame[0].Z());
      TGLVertex3 v2(fYOZProfilePos, fFrame[3].Y(), fFrame[3].Z());
      TGLVertex3 v3(fYOZProfilePos, fFrame[7].Y(), fFrame[7].Z());
      TGLVertex3 v4(fYOZProfilePos, fFrame[4].Y(), fFrame[4].Z());
      if (fSelectionPass)
         EncodeToColor(selectionBase, 5);
      else {
         glDisable(GL_LIGHTING);
         if (fSelectedPlane == 5) {
            TGLEnableGuard blendGuard(GL_BLEND);
            TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
            glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(3.f);
            glColor3d(0., 0.6, 1.);
            DrawQuadOutline(v1, v2, v3, v4);
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
            glLineWidth(1.f);
         }
         glColor3d(0.6, 0.6, 0.6);
      }
      RootGL::DrawQuadFilled(v1, v2, v3, v4, TGLVertex3(1., 0., 0.));
      TGLDisableGuard depth(GL_DEPTH_TEST);
      DrawProfileY();
      if (!fSelectionPass)
         glEnable(GL_LIGHTING);
   }

}

//______________________________________________________________________________
void TGLLegoPainter::MoveDynamicProfile(Int_t px, Int_t py)
{
   //Create dynamic profile using selected plane
   if (fSelectedPlane == 2) {
      if (fFrontPoint == 2) {
         fXOZProfilePos = fFrame[0].Y();//0.;
         fSelectedPlane = 4;
      } else if (!fFrontPoint) {
         fXOZProfilePos = fFrame[2].Y();
         fSelectedPlane = 4;
      } else if (fFrontPoint == 1) {
         fYOZProfilePos = fFrame[0].X();
         fSelectedPlane = 5;
      } else if (fFrontPoint == 3) {
         fYOZProfilePos = fFrame[1].X();
         fSelectedPlane = 5;
      }
   } else if (fSelectedPlane == 3) {
      if (fFrontPoint == 2) {
         fYOZProfilePos = fFrame[0].X();
         fSelectedPlane = 5;
      } else if (!fFrontPoint) {
         fYOZProfilePos = fFrame[1].X();
         fSelectedPlane = 5;
      } else if (fFrontPoint == 1) {
         fXOZProfilePos = fFrame[2].Y();
         fSelectedPlane = 4;
      } else if (fFrontPoint == 3) {
         fXOZProfilePos = fFrame[0].Y();
         fSelectedPlane = 4;
      }
   }

   Double_t mvMatrix[16] = {0.};
   glGetDoublev(GL_MODELVIEW_MATRIX, mvMatrix);
   Double_t prMatrix[16] = {0.};
   glGetDoublev(GL_PROJECTION_MATRIX, prMatrix);
   Double_t winVertex[3] = {0.};//The third is a dummy
   gluProject(
              fSelectedPlane == 5 ? fYOZProfilePos : 0., 
              fSelectedPlane == 4 ? fXOZProfilePos : 0., 
              0., mvMatrix, prMatrix, fViewport, 
              &winVertex[0], &winVertex[1], &winVertex[2]
             );
   winVertex[0] += px - fMousePosition.fX;
   winVertex[1] += py - fMousePosition.fY;
   Double_t newPoint[3] = {0.};//The third is a dummy
   gluUnProject(winVertex[0], winVertex[1], winVertex[2], mvMatrix, prMatrix, fViewport,
                newPoint, newPoint + 1, newPoint + 2);

   if (fSelectedPlane == 4)
      fXOZProfilePos = newPoint[1];
   else
      fYOZProfilePos = newPoint[0];
}

//______________________________________________________________________________
void TGLLegoPainter::DrawProfileX()
{
   const Int_t nX = fBinsX.second - fBinsX.first + 1;
   const Int_t nY = fBinsY.second - fBinsY.first + 1;
   Int_t binY = -1;

   for (Int_t i = 0; i < nY; ++i)
      if (fY[i] <= fXOZProfilePos && fXOZProfilePos <= fY[i + 1]) {
         binY = i;
         break;
      }
   
   if (binY >= 0) {
      binY += fBinsY.first;
      RootGL::TGLEnableGuard blendGuard(GL_BLEND);
      RootGL::TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      /////////////////////Draw grid on a profile plane///////////////////////
      glPushAttrib(GL_LINE_BIT);//[0
      glEnable(GL_LINE_STIPPLE);//[1
      //Dot lines
      const UShort_t stipple = 0x5555;
      glLineStipple(1, stipple);
      glColor3d(0., 0., 0.);
      for (UInt_t i = 0; i < fZLevels.size(); ++i) {
         glBegin(GL_LINES);
         glVertex3d(fFrame[0].X(), fXOZProfilePos, fZLevels[i] * fScaleZ);
         glVertex3d(fFrame[1].X(), fXOZProfilePos, fZLevels[i] * fScaleZ);
         glEnd();
      }
      glDisable(GL_LINE_STIPPLE);//1]
      glPopAttrib();//0]
      //////////////////////////////////////////////////////////////////////////
      glColor3d(1., 0., 0.);
      glLineWidth(3.f);

      for (Int_t i = 0, ir = fBinsX.first; i < nX; ++i, ++ir) {
         Double_t zMax = fHist->GetBinContent(ir, binY);
         if (!ClampZ(zMax))
            continue;

         glBegin(GL_LINE_LOOP);
         glVertex3d(fX[i], fXOZProfilePos, fMinZ);
         glVertex3d(fX[i], fXOZProfilePos, zMax);
         glVertex3d(fX[i + 1], fXOZProfilePos, zMax);
         glVertex3d(fX[i + 1], fXOZProfilePos, fMinZ);
         glEnd();
      }

      glLineWidth(1.f);
   }
}

//______________________________________________________________________________
void TGLLegoPainter::DrawProfileY()
{
   const Int_t nX = fBinsX.second - fBinsX.first + 1;
   const Int_t nY = fBinsY.second - fBinsY.first + 1;
   Int_t binX = -1;

   for (Int_t i = 0; i < nX; ++i)
      if (fX[i] <= fYOZProfilePos && fYOZProfilePos <= fX[i + 1]) {
         binX = i;
         break;
      }
   
   if (binX >= 0) {
      binX += fBinsX.first;
      RootGL::TGLEnableGuard blendGuard(GL_BLEND);
      RootGL::TGLEnableGuard lineSmooth(GL_LINE_SMOOTH);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
      /////////////////////Draw grid on a profile plane///////////////////////
      glPushAttrib(GL_LINE_BIT);//[0
      glEnable(GL_LINE_STIPPLE);//[1
      //Dot lines
      const UShort_t stipple = 0x5555;
      glLineStipple(1, stipple);
      glColor3d(0., 0., 0.);
      for (UInt_t i = 0; i < fZLevels.size(); ++i) {
         glBegin(GL_LINES);
         glVertex3d(fYOZProfilePos, fFrame[0].Y(), fZLevels[i] * fScaleZ);
         glVertex3d(fYOZProfilePos, fFrame[2].Y(), fZLevels[i] * fScaleZ);
         glEnd();
      }
      glDisable(GL_LINE_STIPPLE);//1]
      glPopAttrib();//0]
      //////////////////////////////////////////////////////////////////////////
      glColor3d(1., 0., 0.);
      glLineWidth(3.f);

      for (Int_t i = 0, ir = fBinsY.first; i < nY; ++i, ++ir) {
         Double_t zMax = fHist->GetBinContent(binX, ir);
         if (!ClampZ(zMax))
            continue;

         glBegin(GL_LINE_LOOP);
         glVertex3d(fYOZProfilePos, fY[i], fMinZ);
         glVertex3d(fYOZProfilePos, fY[i], zMax);
         glVertex3d(fYOZProfilePos, fY[i + 1], zMax);
         glVertex3d(fYOZProfilePos, fY[i + 1], fMinZ);
         glEnd();
      }

      glLineWidth(1.f);
   }
}

//______________________________________________________________________________
void TGLLegoPainter::ProcessEvent(Int_t event, Int_t, Int_t)
{
   if (event == kButton1Double && (fXOZProfilePos > fFrame[0].Y() || fYOZProfilePos > fFrame[0].X())) {
      fXOZProfilePos = fFrame[0].Y();
      fYOZProfilePos = fFrame[0].X();
      gGLManager->PaintSingleObject(this);
   }
}

//______________________________________________________________________________
Bool_t TGLLegoPainter::ClampZ(Double_t &zVal)const
{
   if (fLogZ)
      if (zVal <= 0.)
         return kFALSE;
      else 
         zVal = TMath::Log10(zVal) * fScaleZ;
   else 
      zVal *= fScaleZ;

   if (zVal > fFrame[4].Z())
      zVal = fFrame[4].Z();
   else if (zVal < fFrame[0].Z())
      zVal = fFrame[0].Z();

   return kTRUE;
}

//______________________________________________________________________________
void TGLLegoPainter::Enable1DTexture()
{
   //Enable 1D texture.
   //Must be in TGLTexture1D class in TGLUtil.
   glEnable(GL_TEXTURE_1D);
   
   if (!glIsTexture(fTextureName)) {
      glGenTextures(1, &fTextureName);
   }

   glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
   glBindTexture(GL_TEXTURE_1D, fTextureName);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
   glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, fTexture.size() / 4, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, &fTexture[0]);
   glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
}

//______________________________________________________________________________
void TGLLegoPainter::Disable1DTexture()
{
   //Disable 1D texture.
   //Must be in TGLTexture1D class in TGLUtil.
   glDisable(GL_TEXTURE_1D);
   glDeleteTextures(1, &fTextureName);
}

//______________________________________________________________________________
void TGLLegoPainter::DrawGrid(Int_t plane)const
{
   //Grid at XOZ or YOZ back plane
   //Under win32 glPushAttrib does not help with GL_LINE_STIPPLE enable bit
   glPushAttrib(GL_LINE_BIT);//[0
   //Dot lines
   RootGL::TGLEnableGuard stippleGuard(GL_LINE_STIPPLE);//[1-1]
   const UShort_t stipple = 0x5555;
   glLineStipple(1, stipple);
   Double_t lineCaps[][4] = {
      {fFrame[0].X(), fFrame[0].Y(), fFrame[1].X(), fFrame[1].Y()},
      {fFrame[1].X(), fFrame[1].Y(), fFrame[2].X(), fFrame[2].Y()}, 
      {fFrame[2].X(), fFrame[2].Y(), fFrame[3].X(), fFrame[3].Y()},
      {fFrame[0].X(), fFrame[0].Y(), fFrame[3].X(), fFrame[3].Y()}
   };

   for (UInt_t i = 0; i < fZLevels.size(); ++i) {
      glBegin(GL_LINES);
      glVertex3d(lineCaps[plane][0], lineCaps[plane][1], fZLevels[i] * fScaleZ);
      glVertex3d(lineCaps[plane][2], lineCaps[plane][3], fZLevels[i] * fScaleZ);
      glEnd();
   }
 
   glPopAttrib();//0]
}

//______________________________________________________________________________
void TGLLegoPainter::DrawShadow(Int_t plane)const
{
   if (!plane || plane == 2) {
      //XOZ projection.
   } else {
      //YOZ projection.
   }
}
