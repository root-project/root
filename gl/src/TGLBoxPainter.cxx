#include <ctype.h>

#include "Buttons.h"
#include "TColor.h"
#include "TStyle.h"
#include "TMath.h"
#include "TH1.h"

#include "TGLOrthoCamera.h"
#include "TGLBoxPainter.h"
#include "TGLIncludes.h"

ClassImp(TGLBoxPainter)


//______________________________________________________________________________
TGLBoxPainter::TGLBoxPainter(TH1 *hist, TGLOrthoCamera *cam, TGLPlotCoordinates *coord, Int_t ctx)
                  : TGLPlotPainter(hist, cam, coord, ctx, kTRUE),
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
      glColor4d(0., 0., 0., 0.2);
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

   Float_t diffColor[] = {0.8f, 0.8f, 0.8f, 0.2f};

   if (fHist->GetFillColor() != kWhite)
      if (const TColor *c = gROOT->GetColor(fHist->GetFillColor()))
         c->GetRGB(diffColor[0], diffColor[1], diffColor[2]);
   
   glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffColor);
   const Float_t specColor[] = {1.f, 1.f, 1.f, 1.f};
   glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specColor);
   glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 70.f);
}


//______________________________________________________________________________
void TGLBoxPainter::DrawSectionX()const
{
   // Draw XOZ parallel section.

   if (fSelectionPass)
      return;
   const Int_t yBin = fYAxis->FindBin(fXOZSectionPos / fCoord->GetYScale());

   if (yBin && yBin < fYAxis->GetNbins() + 1) {
      glColor4d(0., 0., 0., 0.6);

      for (Int_t ir = fCoord->GetFirstXBin(), e = fCoord->GetLastXBin(); ir <= e; ++ir) {
         for (Int_t kr = fCoord->GetFirstZBin(), e1 = fCoord->GetLastZBin(); kr <= e1; ++kr) {
            Double_t width = fHist->GetBinContent(ir, yBin, kr) / fMinMaxVal.second;
            if (!width)
               continue;

            glBegin(GL_LINE_LOOP);
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 - width * fXAxis->GetBinWidth(ir) / 2),  
                       fXOZSectionPos, fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 - width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 - width * fXAxis->GetBinWidth(ir) / 2),  
                       fXOZSectionPos, fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 + width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 + width * fXAxis->GetBinWidth(ir) / 2), 
                       fXOZSectionPos, fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 + width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 + width * fXAxis->GetBinWidth(ir) / 2), 
                       fXOZSectionPos, fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 - width * fZAxis->GetBinWidth(kr) / 2));
            glEnd();
         }
      }
   }
}


//______________________________________________________________________________
void TGLBoxPainter::DrawSectionY()const
{
   // Draw YOZ parallel section.

   const Int_t xBin = fXAxis->FindBin(fYOZSectionPos / fCoord->GetXScale());

   if (xBin && xBin < fXAxis->GetNbins() + 1) {
      glColor4d(0., 0., 0., 0.6);

      for (Int_t jr = fCoord->GetFirstYBin(), e = fCoord->GetLastYBin(); jr <= e; ++jr) {
         for (Int_t kr = fCoord->GetFirstZBin(), e1 = fCoord->GetLastZBin(); kr <= e1; ++kr) {
            Double_t width = fHist->GetBinContent(xBin, jr, kr) / fMinMaxVal.second;
            if (!width)
               continue;

            glBegin(GL_LINE_LOOP);
            glVertex3d(fYOZSectionPos, fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 - width * fYAxis->GetBinWidth(jr) / 2), 
                       fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 - width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fYOZSectionPos, fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 - width * fYAxis->GetBinWidth(jr) / 2), 
                       fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 + width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fYOZSectionPos, fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 + width * fYAxis->GetBinWidth(jr) / 2), 
                       fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 + width * fZAxis->GetBinWidth(kr) / 2));
            glVertex3d(fYOZSectionPos, fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 + width * fYAxis->GetBinWidth(jr) / 2),
                       fCoord->GetZScale() * (fZAxis->GetBinLowEdge(kr) / 2 + 
                       fZAxis->GetBinUpEdge(kr) / 2 - width * fZAxis->GetBinWidth(kr) / 2));
            glEnd();
         }
      }
   }
}


//______________________________________________________________________________
void TGLBoxPainter::DrawSectionZ()const
{
   // Draw XOY parallel section.

   const Int_t zBin = fZAxis->FindBin(fXOYSectionPos / fCoord->GetZScale());

   if (zBin && zBin < fZAxis->GetNbins() + 1) {
      glColor4d(0., 0., 0., 0.6);

      for (Int_t ir = fCoord->GetFirstXBin(), e = fCoord->GetLastXBin(); ir <= e; ++ir) {
         for (Int_t jr = fCoord->GetFirstYBin(), e1 = fCoord->GetLastYBin(); jr <= e1; ++jr) {
            Double_t width = fHist->GetBinContent(ir, jr, zBin) / fMinMaxVal.second;
            if (!width)
               continue;

            glBegin(GL_LINE_LOOP);
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 + width * fXAxis->GetBinWidth(ir) / 2),
                       fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 + width * fYAxis->GetBinWidth(jr) / 2),
                       fXOYSectionPos);
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 + width * fXAxis->GetBinWidth(ir) / 2),
                       fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 - width * fYAxis->GetBinWidth(jr) / 2),
                       fXOYSectionPos);
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 - width * fXAxis->GetBinWidth(ir) / 2), 
                       fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 - width * fYAxis->GetBinWidth(jr) / 2),
                       fXOYSectionPos);
            glVertex3d(fCoord->GetXScale() * (fXAxis->GetBinLowEdge(ir) / 2 + 
                       fXAxis->GetBinUpEdge(ir) / 2 - width * fXAxis->GetBinWidth(ir) / 2),
                       fCoord->GetYScale() * (fYAxis->GetBinLowEdge(jr) / 2 + 
                       fYAxis->GetBinUpEdge(jr) / 2 + width * fYAxis->GetBinWidth(jr) / 2),
                       fXOYSectionPos);
            glEnd();
         }
      }
   }
}


//______________________________________________________________________________
Bool_t TGLBoxPainter::HasSections()const
{
   // Check, if any section exists.
   
   return fXOZSectionPos > fBackBox.Get3DBox()[0].Y() || fYOZSectionPos> fBackBox.Get3DBox()[0].X() ||
          fXOYSectionPos > fBackBox.Get3DBox()[0].Z();
}
