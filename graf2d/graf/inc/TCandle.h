// @(#)root/graf:$Id$
// Author: Georg Troska 2016/04/14

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TCandle
#define ROOT_TCandle

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif
#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif
#ifndef ROOT_TAttMarket
#include "TAttMarker.h"
#endif

#include "TH1D.h"
#include "TMath.h"

const Int_t kNMAXPOINTS = 2010;  // Max outliers per candle

class TCandle : public TAttLine, public TAttFill, public TAttMarker {
public:
   //Candle Option
   enum CandleOption {
      kNoOption           = 0,
      kBox                = 1,
      kMedianLine         = 10,
      kMedianNotched      = 20,
      kMedianCircle       = 30,
      kMeanLine           = 100,
      kMeanCircle         = 300,
      kWhiskerAll         = 1000,
      kWhisker15          = 2000,
      kAnchor             = 10000,
      kPointsOutliers     = 100000,
      kPointsAll          = 200000,
      kPointsAllScat      = 300000,
      kHistoLeft          = 1000000,
      kHistoRight         = 2000000,
      kHistoViolin        = 3000000,
      kHistoZeroIndicator = 10000000,
      kHorizontal         = 1000000000 // if this bit is not set it is vertical
   };

protected:

   bool fIsRaw;                           ///< 0: for TH1 projection, 1: using raw data
   bool fIsCalculated;
   TH1D * fProj;
   bool fDismiss;                         ///< True if the candle cannot be painted

   Double_t fPosCandleAxis;               ///< x-pos for a vertical candle
   Double_t fCandleWidth;                 ///< The candle width

   Double_t fMean;                        ///< Position of the mean
   Double_t fMedian;                      ///< Position of the median
   Double_t fMedianErr;                   ///< The size of the notch
   Double_t fBoxUp;                       ///< Position of the upper box end
   Double_t fBoxDown;                     ///< Position of the lower box end
   Double_t fWhiskerUp;                   ///< Position of the upper whisker end
   Double_t fWhiskerDown;                 ///< Position of the lower whisker end

   Double_t * fDatapoints;                ///< position of all Datapoints within this candle
   Long64_t fNDatapoints;                 ///< Number of Datapoints within this candle

   Double_t fDrawPointsX[kNMAXPOINTS];    ///< x-coord for every outlier, ..
   Double_t fDrawPointsY[kNMAXPOINTS];    ///< y-coord for every outlier, ..
   Long64_t fNDrawPoints;                 ///< max number of outliers or other point to be shown

   Double_t fHistoPointsX[kNMAXPOINTS];   ///< x-coord for the polyline of the histo
   Double_t fHistoPointsY[kNMAXPOINTS];   ///< y-coord for the polyline of the histo
   int  fNHistoPoints;

   CandleOption fOption;                  ///< Setting the style of the candle
   int fLogX;                             ///< make the candle appear logx-like
   int fLogY;                             ///< make the candle appear logy-like

   Double_t fAxisMin;                     ///< The Minimum which is visible by the axis (used by zero indicator)
   Double_t fAxisMax;                     ///< The Maximum which is visible by the axis (used by zero indicator)

   void Calculate();

   int  GetCandleOption(const int pos) {return (fOption/(int)TMath::Power(10,pos))%10;}
   bool IsOption(CandleOption opt);
   void PaintBox(Int_t nPoints, Double_t *x, Double_t *y, Bool_t swapXY);
   void PaintLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2, Bool_t swapXY);

public:

   TCandle();
   TCandle(const Double_t candlePos, const Double_t candleWidth, Long64_t n, Double_t * points);
   TCandle(const Double_t candlePos, const Double_t candleWidth, TH1D *proj);
   TCandle(const TCandle &candle);
   virtual ~TCandle();

   Double_t       GetMean() const {return fMean;}
   Double_t       GetMedian() const {return fMedian;}
   Double_t       GetQ1() const {return fBoxUp;}
   Double_t       GetQ2() const {return fMedian;}
   Double_t       GetQ3() const {return fBoxDown;}
   Bool_t         IsHorizontal() {return (IsOption(kHorizontal)); }
   Bool_t         IsVertical() {return (!IsOption(kHorizontal)); }

   void           SetOption(CandleOption opt) { fOption = opt; }
   void           SetLog(int x, int y) { fLogX = x; fLogY = y; }
   void           SetAxisPosition(const Double_t candlePos) { fPosCandleAxis = candlePos; }

   void           SetWidth(const Double_t width) { fCandleWidth = width; }
   void           SetHistogram(TH1D *proj) { fProj = proj; fIsCalculated = false;}

   virtual void   Paint(Option_t *option="");
   void           ConvertToPadCoords(Double_t minAxis, Double_t maxAxis, Double_t axisMinCoord, Double_t axisMaxCoord);

   virtual void   SetMean(Double_t mean) { fMean = mean; }
   virtual void   SetMedian(Double_t median) { fMedian = median; }
   virtual void   SetQ1(Double_t q1) { fBoxUp = q1; }
   virtual void   SetQ2(Double_t q2) { fMedian = q2; }
   virtual void   SetQ3(Double_t q3) { fBoxDown = q3; }

   int            ParseOption(char *optin);

   ClassDef(TCandle,1)  //A Candle
};
#endif
