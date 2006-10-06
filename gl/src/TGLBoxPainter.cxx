#include <ctype.h>

#include "Buttons.h"
#include "TColor.h"
#include "TStyle.h"
#include "TMath.h"
#include "TH1.h"
#include "TH3.h"

#include "TGLOrthoCamera.h"
#include "TGLBoxPainter.h"
#include "TGLIncludes.h"

ClassImp(TGLBoxPainter)


//______________________________________________________________________________
TGLBoxPainter::TGLBoxPainter(TH1 *hist, TGLOrthoCamera *cam, TGLPlotCoordinates *coord, Int_t ctx)
                  : TGLPlotPainter(hist, cam, coord, ctx, kTRUE),
                    fXOZSlice("XOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOZ),
                    fYOZSlice("YOZ", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kYOZ),
                    fXOYSlice("XOY", (TH3 *)hist, coord, &fBackBox, TGLTH3Slice::kXOY),
                    fType(kBox)
{
   // Normal constructor.
}


//______________________________________________________________________________
char *TGLBoxPainter::GetPlotInfo(Int_t, Int_t)
{
   //Show box info (i, j, k, binContent).

   fPlotInfo = "";

   if (fSelectedPart) {
      if (fSelectedPart < 6) {
         if (fHist->Class())
            fPlotInfo += fHist->Class()->GetName();
         fPlotInfo += "::";
         fPlotInfo += fHist->GetName();
      } else {
         const Int_t arr2Dsize = fCoord->GetNYBins() * fCoord->GetNZBins();
         const Int_t binI = (fSelectedPart - 6) / arr2Dsize + fCoord->GetFirstXBin();
         const Int_t binJ = (fSelectedPart - 6) % arr2Dsize / fCoord->GetNZBins() + fCoord->GetFirstYBin();
         const Int_t binK = (fSelectedPart - 6) % arr2Dsize % fCoord->GetNZBins() + fCoord->GetFirstZBin();         

         fPlotInfo.Form("(binx = %d; biny = %d; binz = %d; binc = %f)", binI, binJ, binK,
                        fHist->GetBinContent(binI, binJ, binK));
      }
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
   fCamera->SetViewVolume(fBackBox.Get3DBox());

   fMinMaxVal.second  = fHist->GetBinContent(fCoord->GetFirstXBin(), fCoord->GetFirstYBin(), fCoord->GetFirstZBin());
   fMinMaxVal.first = 0.;

   for (Int_t ir = fCoord->GetFirstXBin(); ir <= fCoord->GetLastXBin(); ++ir)
      for (Int_t jr = fCoord->GetFirstYBin(); jr <= fCoord->GetLastYBin(); ++jr)
         for (Int_t kr = fCoord->GetFirstZBin();  kr <= fCoord->GetLastZBin(); ++kr)
            fMinMaxVal.second = TMath::Max(fMinMaxVal.second, fHist->GetBinContent(ir, jr, kr));

   if (!fMinMaxVal.second)
      fMinMaxVal.second = 1.;

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
}


//______________________________________________________________________________
void TGLBoxPainter::Pan(Int_t px, Int_t py)
{
   // User's moving mouse cursor, with middle mouse button pressed (for pad).
   // Calculate 3d shift related to 2d mouse movement.

   if (!MakeGLContextCurrent())
      return;

   if (fSelectedPart > 6)//Pan camera.
      fCamera->Pan(px, py);
   else if (fSelectedPart > 0) {
      //Convert py into bottom-top orientation.
      py = fCamera->GetHeight() - py;
      MoveSection(px, py);
   }

   fMousePosition.fX = px, fMousePosition.fY = py;
   fUpdateSelection = kTRUE;
}


//______________________________________________________________________________
void TGLBoxPainter::AddOption(const TString &option)
{
   // Box1 == spheres.
   
   const Ssiz_t boxPos = option.Index("box");//"box" _already_ _exists_ in a string.
   if (boxPos + 3 < option.Length() && isdigit(option[boxPos + 3]))
      option[boxPos + 3] - '0' == 1 ? fType = kBox1 : fType = kBox;
   else
      fType = kBox;
}


//______________________________________________________________________________
void TGLBoxPainter::ProcessEvent(Int_t event, Int_t /*px*/, Int_t /*py*/)
{
   // Remove sections.

   if (event == kButton1Double && HasSections()) {
      fXOZSectionPos = fBackBox.Get3DBox()[0].Y();
      fYOZSectionPos = fBackBox.Get3DBox()[0].X();
      fXOYSectionPos = fBackBox.Get3DBox()[0].Z();
      gGLManager->PaintSingleObject(this);
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
void TGLBoxPainter::DrawPlot()const
{
   // Draw set of boxes (spheres)

   fBackBox.DrawBox(fSelectedPart, fSelectionPass, fZLevels);
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

   for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
      for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
         for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
            Double_t w = fHist->GetBinContent(ir, jr, kr) / fMinMaxVal.second;
            if (!w)
               continue;

            const Int_t binID = 6 + i * fCoord->GetNZBins() * fCoord->GetNYBins() + j * fCoord->GetNZBins() + k;

            if (fSelectionPass)
               Rgl::ObjectIDToColor(binID);
            else if(fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gOrangeEmission);

               if (fType == kBox)
                  Rgl::DrawBoxFront(xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 - w * xA->GetBinWidth(ir) / 2),
                                    xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 + w * xA->GetBinWidth(ir) / 2),
                                    yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 - w * yA->GetBinWidth(jr) / 2),
                                    yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 + w * yA->GetBinWidth(jr) / 2),
                                    zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 - w * zA->GetBinWidth(kr) / 2),
                                    zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 + w * zA->GetBinWidth(kr) / 2),
                                    frontPoint);
               else
                  Rgl::DrawSphere(&fQuadric,
                                  xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 - w * xA->GetBinWidth(ir) / 2),
                                  xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 + w * xA->GetBinWidth(ir) / 2),
                                  yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 - w * yA->GetBinWidth(jr) / 2),
                                  yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 + w * yA->GetBinWidth(jr) / 2),
                                  zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 - w * zA->GetBinWidth(kr) / 2),
                                  zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 + w * zA->GetBinWidth(kr) / 2)
                                 );

            if (!fSelectionPass && fSelectedPart == binID)
               glMaterialfv(GL_FRONT, GL_EMISSION, Rgl::gNullEmission);
         }
      }
   }


   if (!fSelectionPass && fType != kBox1) {
      glDisable(GL_POLYGON_OFFSET_FILL);//0]
      TGLDisableGuard lightGuard(GL_LIGHTING);//[2 - 2]
      glColor4d(0., 0., 0., 0.4);
      glPolygonMode(GL_FRONT, GL_LINE);//[3

      const TGLEnableGuard blendGuard(GL_BLEND);//[4-4] + 1]
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      const TGLEnableGuard smoothGuard(GL_LINE_SMOOTH);//[5-5]
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);

      for(Int_t ir = irInit, i = iInit; addI > 0 ? i < nX : i >= 0; ir += addI, i += addI) {
         for(Int_t jr = jrInit, j = jInit; addJ > 0 ? j < nY : j >= 0; jr += addJ, j += addJ) {
            for(Int_t kr = krInit, k = kInit; addK > 0 ? k < nZ : k >= 0; kr += addK, k += addK) {
               Double_t w = fHist->GetBinContent(ir, jr, kr) / fMinMaxVal.second;
               if (!w)
                  continue;

               Rgl::DrawBoxFront(xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 - w * xA->GetBinWidth(ir) / 2),
                                 xScale * (xA->GetBinLowEdge(ir) / 2 + xA->GetBinUpEdge(ir) / 2 + w * xA->GetBinWidth(ir) / 2),
                                 yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 - w * yA->GetBinWidth(jr) / 2),
                                 yScale * (yA->GetBinLowEdge(jr) / 2 + yA->GetBinUpEdge(jr) / 2 + w * yA->GetBinWidth(jr) / 2),
                                 zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 - w * zA->GetBinWidth(kr) / 2),
                                 zScale * (zA->GetBinLowEdge(kr) / 2 + zA->GetBinUpEdge(kr) / 2 + w * zA->GetBinWidth(kr) / 2),
                                 frontPoint);
            }
         }
      }  

      glPolygonMode(GL_FRONT, GL_FILL);//3]
   }
}


//______________________________________________________________________________
void TGLBoxPainter::ClearBuffers()const
{
   // Clear buffer.

   Float_t rgb[3] = {1.f, 1.f, 1.f};
   if (const TColor *color = GetPadColor())
      color->GetRGB(rgb[0], rgb[1], rgb[2]);
   glClearColor(rgb[0], rgb[1], rgb[2], 1.);
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}


//______________________________________________________________________________
void TGLBoxPainter::SetPlotColor()const
{
   // Set boxes color.

   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.25f};

   if (fHist->GetFillColor() != kWhite)
      if (const TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   
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

ClassImp(TGLTH3Slice)

//______________________________________________________________________________
TGLTH3Slice::TGLTH3Slice(const TString &name, const TH3 *hist, const TGLPlotCoordinates *coord,
                         const TGLPlotBox *box, ESliceAxis axis)
               : TNamed(name, name),
                 fAxisType(axis),
                 fAxis(0),
                 fCoord(coord),
                 fBox(box),
                 fSliceWidth(1),
                 fHist(hist)
{
   //Ctor.
   fAxis = fAxisType == kXOZ ? fHist->GetYaxis() : fAxisType == kYOZ ? fHist->GetXaxis() : fHist->GetZaxis();
}

//______________________________________________________________________________
void TGLTH3Slice::SetSliceWidth(Int_t width)
{
   // Set Slice width.

   if (width <= 0)
      return;
   
   if (fAxis->GetLast() - fAxis->GetFirst() + 1 <= width)
      fSliceWidth = fAxis->GetLast() - fAxis->GetFirst() + 1;
   else
      fSliceWidth = width;
}

//______________________________________________________________________________
void TGLTH3Slice::DrawSlice(Double_t pos)const
{
   // Draw slice.

   const Int_t bin = fAxis->FindBin(pos);
 //  TGLDisableGuard depthGuard(GL_DEPTH_TEST);

   if (bin && bin < fAxis->GetNbins() + 1) {
      Int_t low = 1, up = 2;
      if (bin - fSliceWidth + 1 >= fAxis->GetFirst()) {
         low = bin - fSliceWidth + 1;
         up  = bin + 1;
      } else {
         low = fAxis->GetFirst();
         up  = bin + (fSliceWidth - (bin - fAxis->GetFirst() + 1)) + 1;
      }
      
      Double_t min = 0., max =0.;
      FindMinMax(min, max, low, up);

      if (!PreparePalette(min, max))
         return;

      PrepareTexCoords();

      if (fPalette.EnableTexture(GL_REPLACE)) {
         const TGLDisableGuard lightGuard(GL_LIGHTING);   
         DrawSliceTextured(pos);
         fPalette.DisableTexture();
         //highlight bins in a slice.
         //DrawSliceFrame(low, up);
      }
   }   
}

void TGLTH3Slice::FindMinMax(Double_t &min, Double_t &max, Int_t low, Int_t up)const
{
   // Find minimum and maximum.

   min = 0.;
   switch (fAxisType) {
   case kXOZ:
      fTexCoords.resize(fCoord->GetNXBins() * fCoord->GetNZBins());
      fTexCoords.SetRowLen(fCoord->GetNXBins());
      for (Int_t level = low; level < up; ++ level)
         min += fHist->GetBinContent(fCoord->GetFirstXBin(), level, fCoord->GetFirstZBin());
      max = min;
      for (Int_t j = fCoord->GetFirstZBin(), jt = 0, ej = fCoord->GetLastZBin(); j <= ej; ++j, ++jt) {
         for (Int_t i = fCoord->GetFirstXBin(), it = 0, ei = fCoord->GetLastXBin(); i <= ei; ++i, ++it) {
            Double_t val = 0.;
            for (Int_t level = low; level < up; ++ level)
               val += fHist->GetBinContent(i, level, j);
            max = TMath::Max(max, val);
            min = TMath::Min(min, val);
            fTexCoords[jt][it] = val;
         }
      }
      break;
   case kYOZ:
      fTexCoords.resize(fCoord->GetNYBins() * fCoord->GetNZBins());
      fTexCoords.SetRowLen(fCoord->GetNYBins());
      for (Int_t level = low; level < up; ++ level)
         min += fHist->GetBinContent(level, fCoord->GetFirstYBin(), fCoord->GetFirstZBin());
      max = min;
      for (Int_t j = fCoord->GetFirstZBin(), jt = 0, ej = fCoord->GetLastZBin(); j <= ej; ++j, ++jt) {
         for (Int_t i = fCoord->GetFirstYBin(), it = 0, ei = fCoord->GetLastYBin(); i <= ei; ++i, ++it) {
            Double_t val = 0.;
            for (Int_t level = low; level < up; ++ level)
               val += fHist->GetBinContent(level, i, j);
            max = TMath::Max(max, val);
            min = TMath::Min(min, val);
            fTexCoords[jt][it] = val;
         }
      }
      break;
   case kXOY:
      fTexCoords.resize(fCoord->GetNXBins() * fCoord->GetNYBins());
      fTexCoords.SetRowLen(fCoord->GetNYBins());
      for (Int_t level = low; level < up; ++ level)
         min += fHist->GetBinContent(fCoord->GetFirstXBin(), fCoord->GetFirstYBin(), level);
      max = min;
      for (Int_t i = fCoord->GetFirstXBin(), ir = 0, ei = fCoord->GetLastXBin(); i <= ei; ++i, ++ir) {
         for (Int_t j = fCoord->GetFirstYBin(), jr = 0, ej = fCoord->GetLastYBin(); j <= ej; ++j, ++jr) {
            Double_t val = 0.;
            for (Int_t level = low; level < up; ++ level)
               val += fHist->GetBinContent(i, j, level);
            max = TMath::Max(max, val);
            min = TMath::Min(min, val);
            fTexCoords[ir][jr] = val;
         }
      }
      break;
   }
}

Bool_t TGLTH3Slice::PreparePalette(Double_t min, Double_t max)const
{
   //Initialize color palette.
   UInt_t paletteSize = ((TH1 *)fHist)->GetContour();
   if (!paletteSize && !(paletteSize = gStyle->GetNumberContours()))
      paletteSize = 20;

   return fPalette.GeneratePalette(paletteSize, Rgl::Range_t(min, max));
}

void TGLTH3Slice::PrepareTexCoords()const
{
   // Prepare TexCoords.

   switch (fAxisType) {
   case kXOZ:
      for (Int_t j = 0, ej = fCoord->GetNZBins(); j < ej; ++j)
         for (Int_t i = 0, ei = fCoord->GetNXBins(); i < ei; ++i)
            fTexCoords[j][i] = fPalette.GetTexCoord(fTexCoords[j][i]);
      break;
   case kYOZ:
      for (Int_t j = 0, ej = fCoord->GetNZBins(); j < ej; ++j)
         for (Int_t i = 0, ei = fCoord->GetNYBins(); i < ei; ++i)
            fTexCoords[j][i] = fPalette.GetTexCoord(fTexCoords[j][i]);
      break;
   case kXOY:
      for (Int_t i = 0, ei = fCoord->GetNXBins(); i < ei; ++i)
         for (Int_t j = 0, ej = fCoord->GetNYBins(); j < ej; ++j)
            fTexCoords[i][j] = fPalette.GetTexCoord(fTexCoords[i][j]);
      break;
   }
}

void TGLTH3Slice::DrawSliceTextured(Double_t pos)const
{
   // Draw slice textured.

   const Double_t xScale = fCoord->GetXScale();
   const Double_t yScale = fCoord->GetYScale();
   const Double_t zScale = fCoord->GetZScale();
   const TAxis *xA = fHist->GetXaxis();
   const TAxis *yA = fHist->GetYaxis();
   const TAxis *zA = fHist->GetZaxis();

   switch (fAxisType) {
   case kXOZ:
      pos *= yScale;
      for (Int_t j = fCoord->GetFirstZBin(), jt = 0, ej = fCoord->GetLastZBin(); j < ej; ++j, ++jt) {
         for (Int_t i = fCoord->GetFirstXBin(), it = 0, ei = fCoord->GetLastXBin(); i < ei; ++i, ++it) {
            const Double_t xMin = xA->GetBinCenter(i) * xScale;
            const Double_t xMax = xA->GetBinCenter(i + 1) * xScale;
            const Double_t zMin = zA->GetBinCenter(j) * zScale;
            const Double_t zMax = zA->GetBinCenter(j + 1) * zScale;
            glBegin(GL_POLYGON);
            glTexCoord1d(fTexCoords[jt][it]);
            glVertex3d(xMin, pos, zMin);
            glTexCoord1d(fTexCoords[jt + 1][it]);
            glVertex3d(xMin, pos, zMax);
            glTexCoord1d(fTexCoords[jt + 1][it + 1]);
            glVertex3d(xMax, pos, zMax);
            glTexCoord1d(fTexCoords[jt][it + 1]);
            glVertex3d(xMax, pos, zMin);
            glEnd();
         }
      }
      break;
   case kYOZ:
      pos *= xScale;
      for (Int_t j = fCoord->GetFirstZBin(), jt = 0, ej = fCoord->GetLastZBin(); j < ej; ++j, ++jt) {
         for (Int_t i = fCoord->GetFirstYBin(), it = 0, ei = fCoord->GetLastYBin(); i < ei; ++i, ++it) {
            const Double_t yMin = yA->GetBinCenter(i) * yScale;
            const Double_t yMax = yA->GetBinCenter(i + 1) * yScale;
            const Double_t zMin = zA->GetBinCenter(j) * zScale;
            const Double_t zMax = zA->GetBinCenter(j + 1) * zScale;
            glBegin(GL_POLYGON);
            glTexCoord1d(fTexCoords[jt][it]);
            glVertex3d(pos, yMin, zMin);
            glTexCoord1d(fTexCoords[jt][it + 1]);
            glVertex3d(pos, yMax, zMin);
            glTexCoord1d(fTexCoords[jt + 1][it + 1]);
            glVertex3d(pos, yMax, zMax);
            glTexCoord1d(fTexCoords[jt + 1][it]);
            glVertex3d(pos, yMin, zMax);
            glEnd();
         }
      }
      break;
   case kXOY:
      pos *= zScale;
      for (Int_t j = fCoord->GetFirstXBin(), jt = 0, ej = fCoord->GetLastXBin(); j < ej; ++j, ++jt) {
         for (Int_t i = fCoord->GetFirstYBin(), it = 0, ei = fCoord->GetLastYBin(); i < ei; ++i, ++it) {
            const Double_t xMin = xA->GetBinCenter(j) * xScale;
            const Double_t xMax = xA->GetBinCenter(j + 1) * xScale;
            const Double_t yMin = yA->GetBinCenter(i) * yScale;
            const Double_t yMax = yA->GetBinCenter(i + 1) * yScale;
            glBegin(GL_POLYGON);
            glTexCoord1d(fTexCoords[jt + 1][it]);
            glVertex3d(xMax, yMin, pos);
            glTexCoord1d(fTexCoords[jt + 1][it + 1]);
            glVertex3d(xMax, yMax, pos);
            glTexCoord1d(fTexCoords[jt][it + 1]);
            glVertex3d(xMin, yMax, pos);
            glTexCoord1d(fTexCoords[jt][it]);
            glVertex3d(xMin, yMin, pos);
            glEnd();
         }
      }
      break;
   }
}

namespace {

   void DrawBoxOutline(Double_t xMin, Double_t xMax, Double_t yMin, 
                       Double_t yMax, Double_t zMin, Double_t zMax)
   {
      glBegin(GL_LINE_LOOP);
      glVertex3d(xMin, yMin, zMin);
      glVertex3d(xMax, yMin, zMin);
      glVertex3d(xMax, yMax, zMin);
      glVertex3d(xMin, yMax, zMin);
      glEnd();

      glBegin(GL_LINE_LOOP);
      glVertex3d(xMin, yMin, zMax);
      glVertex3d(xMax, yMin, zMax);
      glVertex3d(xMax, yMax, zMax);
      glVertex3d(xMin, yMax, zMax);
      glEnd();

      glBegin(GL_LINES);
      glVertex3d(xMin, yMin, zMin);
      glVertex3d(xMin, yMin, zMax);
      glVertex3d(xMax, yMin, zMin);
      glVertex3d(xMax, yMin, zMax);
      glVertex3d(xMax, yMax, zMin);
      glVertex3d(xMax, yMax, zMax);
      glVertex3d(xMin, yMax, zMin);
      glVertex3d(xMin, yMax, zMax);
      glEnd();
   }

}

void TGLTH3Slice::DrawSliceFrame(Int_t low, Int_t up)const
{
   // Draw slice frame.

   glColor3d(1., 0., 0.);
   const TGLVertex3 *box = fBox->Get3DBox();
   
   switch (fAxisType) {
   case kXOZ:
      DrawBoxOutline(box[0].X(), box[1].X(),
                     fAxis->GetBinLowEdge(low) * fCoord->GetYScale(), 
                     fAxis->GetBinUpEdge(up - 1) * fCoord->GetYScale(),
                     box[0].Z(), box[4].Z());
      break;
   case kYOZ:
      DrawBoxOutline(fAxis->GetBinLowEdge(low) * fCoord->GetXScale(), 
                     fAxis->GetBinUpEdge(up - 1) * fCoord->GetXScale(),
                     box[0].Y(), box[2].Y(),
                     box[0].Z(), box[4].Z());
      break;
   case kXOY:
      DrawBoxOutline(box[0].X(), box[1].X(),
                     box[0].Y(), box[2].Y(),
                     fAxis->GetBinLowEdge(low) * fCoord->GetZScale(), 
                     fAxis->GetBinUpEdge(up - 1) * fCoord->GetZScale());
      break;
   }
}
