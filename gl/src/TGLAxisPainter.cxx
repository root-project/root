// @(#)root/gl:$Name:  $:$Id: TGLAxisPainter.cxx,v 1.1 2006/06/14 10:00:00 couet Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <string>

#include "THLimitsFinder.h"
#include "TVirtualPad.h"
#include "TVirtualGL.h"
#include "TVirtualX.h"
#include "TGaxis.h"
#include "TMath.h"
#include "TAxis.h"
#include "TH1.h"

#include "TGLAxisPainter.h"
#include "TGLPlotPainter.h"

ClassImp(TGLAxisPainter)

//______________________________________________________________________________
TGLAxisPainter::~TGLAxisPainter()
{
   // TGLAxisPainter destructor.
}

ClassImp(TGL2DAxisPainter)

//______________________________________________________________________________
TGL2DAxisPainter::TGL2DAxisPainter(TH1 *hist)
                     : fPlotFrame(0),
                       fAxisX(hist->GetXaxis()),
                       fAxisY(hist->GetYaxis()),
                       fAxisZ(hist->GetZaxis())
{
   //TGL2DAxisPainter constructor.
}

//______________________________________________________________________________
void TGL2DAxisPainter::SetPlotFrame(TGLPlotFrame *frame)
{
   // Set plot frame.
   fPlotFrame = frame;
}

//______________________________________________________________________________
void TGL2DAxisPainter::SetRanges(const Range_t &xRange, const Range_t &yRange, const Range_t &zRange)
{
   // Set range.
   fRangeX = xRange;
   fRangeY = yRange;
   fRangeZ = zRange;
}

//______________________________________________________________________________
void TGL2DAxisPainter::SetZLevels(std::vector<Double_t> &zLevels)
{
   //Define levels for grid.
   //copy vals, because Optimize can chnage them.
   Double_t zMin = fRangeZ.first;
   Double_t zMax = fRangeZ.second;
   Int_t nDiv = fAxisZ->GetNdivisions() % 100;
   Int_t nBins = 0;
   Double_t binLow = 0., binHigh = 0., binWidth = 0.;
   
   THLimitsFinder::Optimize(zMin, zMax, nDiv, binLow, binHigh, nBins, binWidth, " ");
   zLevels.resize(nBins + 1);
   
   for (Int_t i = 0; i < nBins + 1; ++i)
      zLevels[i] = binLow + i * binWidth;
}

namespace {

   void Draw2DAxis(TAxis *axis, Double_t xMin, Double_t yMin, Double_t xMax, Double_t yMax,
                   Double_t min, Double_t max, Bool_t log, Bool_t z = kFALSE)
   {
      //Axes are drawn with help of TGaxis class
      std::string option;
      option.reserve(20);
      
      if (xMin > xMax || z) option += "SDH=+";
      else option += "SDH=-";
      
      if (log) option += 'G';
      
      Int_t nDiv = axis->GetNdivisions();
      
      if (nDiv < 0) {
         option += 'N';
         nDiv = -nDiv;
      }
      
      TGaxis axisPainter;
      axisPainter.SetLineWidth(1);
      
      static const Double_t zero = 0.001;
      
      if (TMath::Abs(xMax - xMin) >= zero || TMath::Abs(yMax - yMin) >= zero) {
         axisPainter.ImportAxisAttributes(axis);
         axisPainter.SetLabelOffset(axis->GetLabelOffset() + axis->GetTickLength());

         if (log) {
            min = TMath::Power(10, min);
            max = TMath::Power(10, max);
         }
         //Option time display is required ?
         if (axis->GetTimeDisplay()) {
            option += 't';

            if (!strlen(axis->GetTimeFormatOnly()))
               axisPainter.SetTimeFormat(axis->ChooseTimeFormat(max - min));
            else
               axisPainter.SetTimeFormat(axis->GetTimeFormat());
         }

         axisPainter.SetOption(option.c_str());
         axisPainter.PaintAxis(xMin, yMin, xMax, yMax, min, max, nDiv, option.c_str());
      }
   }

   const Int_t gFramePoints[][2] = {{3, 1}, {0, 2}, {1, 3}, {2, 0}};
   //Each point has two "neighbouring axes" (left and right). Axes types are 1 (ordinata) and 0 (abscissa)
   const Int_t gAxisType[][2] = {{1, 0}, {0, 1}, {1, 0}, {0, 1}};

}

//______________________________________________________________________________
void TGL2DAxisPainter::Paint(Int_t glContext)
{
   //Using front point, find, where to draw axes and which labels to use for them
   gGLManager->SelectOffScreenDevice(glContext);
   gVirtualX->SetDrawMode(TVirtualX::kCopy);

   const Int_t left = gFramePoints[fPlotFrame->fFrontPoint][0];
   const Int_t right = gFramePoints[fPlotFrame->fFrontPoint][1];
   const Double_t xLeft = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() 
                                            + fPlotFrame->f2DAxes[left].X()));
   const Double_t yLeft = gPad->AbsPixeltoY(Int_t(fPlotFrame->fViewport[3] - fPlotFrame->f2DAxes[left].Y()
                                            + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() 
                                            + fPlotFrame->fViewport[1]));
   const Double_t xMid = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() 
                                           + fPlotFrame->f2DAxes[fPlotFrame->fFrontPoint].X()));
   const Double_t yMid = gPad->AbsPixeltoY(Int_t(fPlotFrame->fViewport[3] - fPlotFrame->f2DAxes[fPlotFrame->fFrontPoint].Y() 
                                           + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() 
                                           + fPlotFrame->fViewport[1]));
   const Double_t xRight = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() 
                                             + fPlotFrame->f2DAxes[right].X()));
   const Double_t yRight = gPad->AbsPixeltoY(Int_t(fPlotFrame->fViewport[3] - fPlotFrame->f2DAxes[right].Y() 
                                             + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() 
                                             + fPlotFrame->fViewport[1]));
   
   const Double_t points[][2] = {
                                 {fRangeX.first, fRangeY.first}, 
                                 {fRangeX.second, fRangeY.first}, 
                                 {fRangeX.second, fRangeY.second}, 
                                 {fRangeX.first, fRangeY.second}
                                 };

   const Int_t leftType = gAxisType[fPlotFrame->fFrontPoint][0];
   const Int_t rightType = gAxisType[fPlotFrame->fFrontPoint][1];
   const Double_t leftLabel = points[left][leftType];
   const Double_t leftMidLabel = points[fPlotFrame->fFrontPoint][leftType];
   const Double_t rightMidLabel = points[fPlotFrame->fFrontPoint][rightType];
   const Double_t rightLabel = points[right][rightType];

   if (xLeft - xMid || yLeft - yMid) {//To supress error messages from TGaxis
      TAxis *axis = leftType ? fAxisY : fAxisX;

      if (leftLabel < leftMidLabel)
         Draw2DAxis(axis, xLeft, yLeft, xMid, yMid, leftLabel, leftMidLabel, leftType ? fPlotFrame->fLogY : fPlotFrame->fLogX);
      else
         Draw2DAxis(axis, xMid, yMid, xLeft, yLeft, leftMidLabel, leftLabel, leftType ? fPlotFrame->fLogY : fPlotFrame->fLogX);
   }

   if (xRight - xMid || yRight - yMid) {//To supress error messages from TGaxis
      TAxis *axis = rightType ? fAxisY : fAxisX;

      if (rightMidLabel < rightLabel)
         Draw2DAxis(axis, xMid, yMid, xRight, yRight, rightMidLabel, rightLabel, rightType ? fPlotFrame->fLogY : fPlotFrame->fLogX);
      else
         Draw2DAxis(axis, xRight, yRight, xMid, yMid, rightLabel, rightMidLabel, rightType ? fPlotFrame->fLogY : fPlotFrame->fLogX);
   }
    
   const Double_t xUp = gPad->AbsPixeltoX(Int_t(gPad->GetXlowNDC() * gPad->GetWw() 
                                          + fPlotFrame->f2DAxes[left + 4].X()));
   const Double_t yUp = gPad->AbsPixeltoY(Int_t(fPlotFrame->fViewport[3] - fPlotFrame->f2DAxes[left + 4].Y() 
                                          + (1 - gPad->GetHNDC() - gPad->GetYlowNDC()) * gPad->GetWh() 
                                          + fPlotFrame->fViewport[1]));
   Draw2DAxis(fAxisZ, xLeft, yLeft, xUp, yUp, fRangeZ.first, fRangeZ.second, fPlotFrame->fLogZ, kTRUE);
   
   gVirtualX->SelectWindow(gPad->GetPixmapID());
}
