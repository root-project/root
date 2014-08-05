#include <algorithm>

#include "KeySymbols.h"
#include "TVirtualX.h"
#include "Buttons.h"
#include "TString.h"
#include "TROOT.h"
#include "TClass.h"
#include "TColor.h"
#include "TStyle.h"
#include "TH3.h"
#include "TF1.h"

#include "TGLVoxelPainter.h"
#include "TGLPlotCamera.h"
#include "TGLIncludes.h"

//______________________________________________________________________________
//
// Paint TH3 histograms as "voxels" - colored boxes, transparent if transfer function was specified.
//

ClassImp(TGLVoxelPainter)

//______________________________________________________________________________
TGLVoxelPainter::TGLVoxelPainter(TH1 *hist, TGLPlotCamera *cam, TGLPlotCoordinates *coord)
                  : TGLPlotPainter(hist, cam, coord, kFALSE, kFALSE, kFALSE),
                    fTransferFunc(0)
{
   // Constructor.
   //This plot always needs a palette.
   fDrawPalette = kTRUE;
}


//______________________________________________________________________________
char *TGLVoxelPainter::GetPlotInfo(Int_t, Int_t)
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
Bool_t TGLVoxelPainter::InitGeometry()
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

   if (fCoord->Modified()) {
      fUpdateSelection = kTRUE;
      fXOZSectionPos = fBackBox.Get3DBox()[0].Y();
      fYOZSectionPos = fBackBox.Get3DBox()[0].X();
      fXOYSectionPos = fBackBox.Get3DBox()[0].Z();
      fCoord->ResetModified();
   }

   const TList *funcList = fHist->GetListOfFunctions();
   fTransferFunc = dynamic_cast<TF1*>(funcList->FindObject("TransferFunction"));

   return kTRUE;
}


//______________________________________________________________________________
void TGLVoxelPainter::StartPan(Int_t px, Int_t py)
{
   // User clicks right mouse button (in a pad).

   fMousePosition.fX = px;
   fMousePosition.fY = fCamera->GetHeight() - py;
   fCamera->StartPan(px, py);
   fBoxCut.StartMovement(px, fCamera->GetHeight() - py);
}


//______________________________________________________________________________
void TGLVoxelPainter::Pan(Int_t px, Int_t py)
{
   // User's moving mouse cursor, with middle mouse button pressed (for pad).
   // Calculate 3d shift related to 2d mouse movement.

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
void TGLVoxelPainter::AddOption(const TString &option)
{
   // "z" draw palette or not.
   option.Index("z") == kNPOS ? fDrawPalette = kFALSE : fDrawPalette = kTRUE;

}

//______________________________________________________________________________
void TGLVoxelPainter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t py)
{
   // Remove sections, switch on/off box cut.

   if (event == kButton1Double && fBoxCut.IsActive()) {
      if (fBoxCut.IsActive())
         fBoxCut.TurnOnOff();
      if (!gVirtualX->IsCmdThread())
         gROOT->ProcessLineFast(Form("((TGLPlotPainter *)0x%lx)->Paint()", ULong_t(this)));
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
void TGLVoxelPainter::InitGL()const
{
   // Initialize some gl state variables.

   glEnable(GL_DEPTH_TEST);
   glEnable(GL_LIGHTING);
   glEnable(GL_LIGHT0);

   glEnable(GL_CULL_FACE);
   glCullFace(GL_BACK);

   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
}

//______________________________________________________________________________
void TGLVoxelPainter::DeInitGL()const
{
   // Return back some gl state variables.

   glDisable(GL_DEPTH_TEST);
   glDisable(GL_LIGHTING);
   glDisable(GL_LIGHT0);
   glDisable(GL_CULL_FACE);
   glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
}

//______________________________________________________________________________
void TGLVoxelPainter::DrawPlot()const
{
   // Draw "voxels".

   //Shift plot to point of origin.
   const Rgl::PlotTranslation trGuard(this);

   if (!fSelectionPass)
      PreparePalette();

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels, fHighColor);

   TGLDisableGuard depthTest(GL_DEPTH_TEST);

   if (!fSelectionPass) {
      glEnable(GL_BLEND);//[1
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
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
   const Int_t addK = fBackBox.Get2DBox()[frontPoint + 4].Y() > fBackBox.Get2DBox()[frontPoint].Y() ? 1
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

   Float_t rgba[4] = {};

   for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
      for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
//         for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
         for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
            const Double_t xMin = xScale * xA->GetBinLowEdge(ir);
            const Double_t xMax = xScale * xA->GetBinUpEdge(ir);
            const Double_t yMin = yScale * yA->GetBinLowEdge(jr);
            const Double_t yMax = yScale * yA->GetBinUpEdge(jr);
            const Double_t zMin = zScale * zA->GetBinLowEdge(kr);
            const Double_t zMax = zScale * zA->GetBinUpEdge(kr);

            if (fBoxCut.IsActive() && fBoxCut.IsInCut(xMin, xMax, yMin, yMax, zMin, zMax))
               continue;

            FindVoxelColor(fHist->GetBinContent(ir, jr, kr), rgba);

            if (rgba[3] < 0.01f)
               continue;

            if (!fSelectionPass)
               SetVoxelColor(rgba);

            const Int_t binID = fSelectionBase + i * fCoord->GetNZBins() * fCoord->GetNYBins() + j * fCoord->GetNZBins() + k;

            if (fSelectionPass && !fHighColor)
               Rgl::ObjectIDToColor(binID, fHighColor);
            else if(!fHighColor && fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);

            Rgl::DrawBoxFront(xMin, xMax, yMin, yMax, zMin, zMax, frontPoint);

            if (!fSelectionPass && !fHighColor && fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);
         }
      }
   }

   if (fBoxCut.IsActive())
      fBoxCut.DrawBox(fSelectionPass, fSelectedPart);

   if (!fSelectionPass) {
      if (fDrawPalette)
         DrawPalette();
      glDisable(GL_BLEND);//1]
   }
}

//______________________________________________________________________________
void TGLVoxelPainter::DrawSectionXOZ()const
{
   // Noop.
}

//______________________________________________________________________________
void TGLVoxelPainter::DrawSectionYOZ()const
{
   // Noop.
}


//______________________________________________________________________________
void TGLVoxelPainter::DrawSectionXOY()const
{
   // Noop.
}

//______________________________________________________________________________
void TGLVoxelPainter::DrawPalette()const
{
   //Draw. Palette.
   if (!fPalette.GetPaletteSize() || !fCamera)
      return;

   if (!fHist->TestBit(TH1::kUserContour))
      Rgl::DrawPalette(fCamera, fPalette);
   else
      Rgl::DrawPalette(fCamera, fPalette, fLevels);

   glFinish();

   fCamera->SetCamera();
   fCamera->Apply(fPadPhi, fPadTheta);
}

//______________________________________________________________________________
void TGLVoxelPainter::DrawPaletteAxis()const
{
   //Draw. Palette. Axis.
   if (fCamera) {
      gVirtualX->SetDrawMode(TVirtualX::kCopy);//TCanvas by default sets in kInverse
      Rgl::DrawPaletteAxis(fCamera, fMinMaxVal, kFALSE);
   }
}

//______________________________________________________________________________
void TGLVoxelPainter::PreparePalette()const
{
   //Generate palette.
   if(fMinMaxVal.first == fMinMaxVal.second)
      return;//must be std::abs(fMinMaxVal.second - fMinMaxVal.first) < ...

   fLevels.clear();
   UInt_t paletteSize = 0;

   if (fHist->TestBit(TH1::kUserContour)) {
      if (const UInt_t trySize = fHist->GetContour()) {
         fLevels.reserve(trySize);

         for (UInt_t i = 0; i < trySize; ++i) {
            const Double_t level = fHist->GetContourLevel(Int_t(i));
            if (level <= fMinMaxVal.first || level >= fMinMaxVal.second)
               continue;
            fLevels.push_back(level);
         }
         //sort levels
         if (fLevels.size()) {
            std::sort(fLevels.begin(), fLevels.end());
            fLevels.push_back(fMinMaxVal.second);
            fLevels.insert(fLevels.begin(), fMinMaxVal.first);
            fPalette.SetContours(&fLevels);
            paletteSize = fLevels.size() - 1;
         }
      }

      if (!paletteSize)
         fHist->ResetBit(TH1::kUserContour);
   }

   if (!paletteSize && !(paletteSize = gStyle->GetNumberContours()))
      paletteSize = 20;

   fPalette.GeneratePalette(paletteSize, fMinMaxVal);
}

//______________________________________________________________________________
void TGLVoxelPainter::FindVoxelColor(Double_t binContent, Float_t *rgba)const
{
   // Find box color.
   const UChar_t * tc = fPalette.GetColour(binContent);
   rgba[3] = 0.06f; //Just a constant transparency.


   if (fTransferFunc) {
      rgba[3] = fTransferFunc->Eval(binContent);
   }

   rgba[0] = tc[0] / 255.f;
   rgba[1] = tc[1] / 255.f;
   rgba[2] = tc[2] / 255.f;
}


//______________________________________________________________________________
void TGLVoxelPainter::SetVoxelColor(const Float_t *diffColor)const
{
   // Set box color.
   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}
