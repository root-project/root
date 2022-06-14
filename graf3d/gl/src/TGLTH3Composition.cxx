// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  07/08/2009

#include <stdexcept>

#include "KeySymbols.h"
#include "TVirtualX.h"
#include "Buttons.h"
#include "TString.h"
#include "TError.h"
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"

#include "TGLTH3Composition.h"
#include "TGLIncludes.h"

/** \class TGLTH3Composition
\ingroup opengl
*/

ClassImp(TGLTH3Composition);

////////////////////////////////////////////////////////////////////////////////
///I have to define it, since explicit copy ctor was declared.

TGLTH3Composition::TGLTH3Composition()
{
}

namespace {

void CompareAxes(const TAxis *a1, const TAxis *a2, const TString &axisName);

}

////////////////////////////////////////////////////////////////////////////////
///Add TH3 into collection. Throw if fHists is not empty
///but ranges are not equal.

void TGLTH3Composition::AddTH3(const TH3 *h, ETH3BinShape shape)
{
   const TAxis *xa = h->GetXaxis();
   const TAxis *ya = h->GetYaxis();
   const TAxis *za = h->GetZaxis();

   if (!fHists.size()) {
      //This is the first hist in a composition,
      //take its ranges and reset axes for the composition.
      fXaxis.Set(h->GetNbinsX(), xa->GetBinLowEdge(xa->GetFirst()), xa->GetBinUpEdge(xa->GetLast()));
      fYaxis.Set(h->GetNbinsY(), ya->GetBinLowEdge(ya->GetFirst()), ya->GetBinUpEdge(ya->GetLast()));
      fZaxis.Set(h->GetNbinsZ(), za->GetBinLowEdge(za->GetFirst()), za->GetBinUpEdge(za->GetLast()));
   } else {
      CompareAxes(xa, GetXaxis(), "X");
      CompareAxes(ya, GetYaxis(), "Y");
      CompareAxes(za, GetZaxis(), "Z");
   }

   fHists.push_back(TH3Pair_t(h, shape));
}

////////////////////////////////////////////////////////////////////////////////
///Check if "this" is under cursor.

Int_t TGLTH3Composition::DistancetoPrimitive(Int_t px, Int_t py)
{
   if (!fPainter.get())
      return 9999;

   return fPainter->DistancetoPrimitive(px, py);
}

////////////////////////////////////////////////////////////////////////////////
///Mouse and keyboard events.

void TGLTH3Composition::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   fPainter->ExecuteEvent(event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
///I cannot show bin content in a status bar -
///since there can be several bins in one.

char *TGLTH3Composition::GetObjectInfo(Int_t /*px*/, Int_t /*py*/) const
{
   static char message[] = "TH3 composition";
   return message;
}

////////////////////////////////////////////////////////////////////////////////
///Paint a composition of 3d hists.

void TGLTH3Composition::Paint(Option_t * /*option*/)
{
   if (!fHists.size())
      return;

   //create a painter.
   if (!fPainter.get())
      fPainter.reset(new TGLHistPainter(this));

   fPainter->Paint("dummy");
}

/** \class TGLTH3CompositionPainter
\ingroup opengl
*/

ClassImp(TGLTH3CompositionPainter);

////////////////////////////////////////////////////////////////////////////////
///Ctor.

TGLTH3CompositionPainter::TGLTH3CompositionPainter(TGLTH3Composition *data, TGLPlotCamera *cam,
                                                   TGLPlotCoordinates *coord)
                             : TGLPlotPainter(data, cam, coord, kFALSE, kFALSE, kFALSE),
                               fData(data)
{
}

////////////////////////////////////////////////////////////////////////////////
///Will be never called from TPad.

char *TGLTH3CompositionPainter::GetPlotInfo(Int_t /*px*/, Int_t /*py*/)
{
   static char message[] = "TH3 composition";
   return message;
}

////////////////////////////////////////////////////////////////////////////////

Bool_t TGLTH3CompositionPainter::InitGeometry()
{
   if (!fData->fHists.size())
      return kFALSE;

   //Prepare plot painter.
   //Forget about log scale.
   fCoord->SetZLog(kFALSE);
   fCoord->SetYLog(kFALSE);
   fCoord->SetXLog(kFALSE);

   if (!fCoord->SetRanges(fHist, kFALSE, kTRUE))//kFALSE == drawErrors, kTRUE == zAsBins
      return kFALSE;

   fBackBox.SetPlotBox(fCoord->GetXRangeScaled(), fCoord->GetYRangeScaled(), fCoord->GetZRangeScaled());
   if (fCamera)
      fCamera->SetViewVolume(fBackBox.Get3DBox());

   //Loop on hists.
   const TH3 *h = fData->fHists[0].first;
   fMinMaxVal.second  = h->GetBinContent(fCoord->GetFirstXBin(),
                                         fCoord->GetFirstYBin(),
                                         fCoord->GetFirstZBin());
   fMinMaxVal.first = fMinMaxVal.second;

   for (UInt_t hNum = 0, lastH = fData->fHists.size(); hNum < lastH; ++hNum) {
      h = fData->fHists[hNum].first;
      for (Int_t ir = fCoord->GetFirstXBin(); ir <= fCoord->GetLastXBin(); ++ir) {
         for (Int_t jr = fCoord->GetFirstYBin(); jr <= fCoord->GetLastYBin(); ++jr) {
            for (Int_t kr = fCoord->GetFirstZBin();  kr <= fCoord->GetLastZBin(); ++kr) {
               fMinMaxVal.second = TMath::Max(fMinMaxVal.second, h->GetBinContent(ir, jr, kr));
               fMinMaxVal.first = TMath::Min(fMinMaxVal.first, h->GetBinContent(ir, jr, kr));
            }
         }
      }
   }

   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      fCoord->ResetModified();
   }

   return kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
///Move plot or box cut.

void TGLTH3CompositionPainter::StartPan(Int_t px, Int_t py)
{
   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}

////////////////////////////////////////////////////////////////////////////////
/// User's moving mouse cursor, with middle mouse button pressed (for pad).
/// Calculate 3d shift related to 2d mouse movement.

void TGLTH3CompositionPainter::Pan(Int_t px, Int_t py)
{
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
      }

      RestoreProjectionMatrix();
      RestoreModelviewMatrix();
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
///No options for composition.

void TGLTH3CompositionPainter::AddOption(const TString &/*option*/)
{
}

////////////////////////////////////////////////////////////////////////////////
///Switch on/off box cut.

void TGLTH3CompositionPainter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{
   if (event == kButton1Double && fBoxCut.IsActive()) {
      fBoxCut.TurnOnOff();
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLPlotPainter *)0x%zx)->Paint()", (size_t)this));
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

////////////////////////////////////////////////////////////////////////////////
/// Initialize some gl state variables.

void TGLTH3CompositionPainter::InitGL()const
{
   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);
   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

////////////////////////////////////////////////////////////////////////////////
///Return back some gl state variables.

void TGLTH3CompositionPainter::DeInitGL()const
{
   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glDisable(GL_LIGHT0);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

////////////////////////////////////////////////////////////////////////////////
///Draw composition of TH3s.

void TGLTH3CompositionPainter::DrawPlot()const
{
   //Shift plot to point of origin.
   const Rgl::PlotTranslation trGuard(this);

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   if (!fSelectionPass) {
      glEnable(GL_POLYGON_OFFSET_FILL);//[0
      glPolygonOffset(1.f, 1.f);
   } else
      return;

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

   Double_t maxContent = TMath::Max(TMath::Abs(fMinMaxVal.first), TMath::Abs(fMinMaxVal.second));
   if(!maxContent)//bad, find better way to check zero.
      maxContent = 1.;

   for (UInt_t hNum = 0; hNum < fData->fHists.size(); ++hNum) {
      const TH3 *h = fData->fHists[hNum].first;
      const TGLTH3Composition::ETH3BinShape shape = fData->fHists[hNum].second;
      SetColor(h->GetFillColor());

      for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
         for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
            for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
               const Double_t binContent = h->GetBinContent(ir, jr, kr);
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

               if (shape == TGLTH3Composition::kSphere)
                  Rgl::DrawSphere(&fQuadric, xMin, xMax, yMin, yMax, zMin, zMax);
               else
                  Rgl::DrawBoxFront(xMin, xMax, yMin, yMax, zMin, zMax, frontPoint);
            }
         }
      }
   }

   if (fBoxCut.IsActive())
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);

   glDisable(GL_POLYGON_OFFSET_FILL);//0]
   const TGLDisableGuard lightGuard(GL_LIGHTING);//[2 - 2]
   glColor4d(0., 0., 0., 0.25);
   glPolygonMode(GL_FRONT, GL_LINE);//[3

   const TGLEnableGuard blendGuard(GL_BLEND);//[4-4] + 1]
   glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   const TGLEnableGuard smoothGuard(GL_LINE_SMOOTH);//[5-5]
   glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

   for (UInt_t hNum = 0; hNum < fData->fHists.size(); ++hNum) {
      if (fData->fHists[hNum].second == TGLTH3Composition::kSphere)
         continue;//No outlines for spherical bins.

      const TH3 *h = fData->fHists[hNum].first;

      for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
         for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
            for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
               const Double_t w = TMath::Abs(h->GetBinContent(ir, jr, kr)) / maxContent;
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
   }

   glPolygonMode(GL_FRONT, GL_FILL);//3]
}

////////////////////////////////////////////////////////////////////////////////
///Set material.

void TGLTH3CompositionPainter::SetColor(Int_t color)const
{
   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.05f};

   if (color != kWhite)
      if (const TColor *c = gROOT->GetColor(color))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);

   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}

namespace {

////////////////////////////////////////////////////////////////////////////////

void AxisError(const TString & errorMsg)
{
   Error("TGLTH3Composition::AddTH3", "%s", errorMsg.Data());
   throw std::runtime_error(errorMsg.Data());
}

////////////////////////////////////////////////////////////////////////////////
///Check number of bins.

void CompareAxes(const TAxis *a1, const TAxis *a2, const TString &axisName)
{
   if (a1->GetNbins() != a2->GetNbins())
      AxisError("New hist has different number of bins along " + axisName);

   //Check bin ranges.
   const Int_t firstBin1 = a1->GetFirst(), lastBin1 = a1->GetLast();
   const Int_t firstBin2 = a2->GetFirst(), lastBin2 = a2->GetLast();

   if (firstBin1 != firstBin2)
      AxisError("New hist has different first bin along " + axisName);

   if (lastBin1 != lastBin2)
      AxisError("New hist has different last bin along " + axisName);

   const Double_t eps = 1e-7;//?????:((((
   //Check axes ranges.
   if (TMath::Abs(a1->GetBinLowEdge(firstBin1) - a2->GetBinLowEdge(firstBin2)) > eps)
      AxisError("New hist has different low edge along " + axisName);
   if (TMath::Abs(a1->GetBinUpEdge(lastBin1) - a2->GetBinUpEdge(lastBin2)) > eps)
      AxisError("New hist has different low edge along " + axisName);
}

}
