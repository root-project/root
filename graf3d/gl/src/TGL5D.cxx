// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <stdexcept>

#include "TTreeFormula.h"
#include "TTree.h"
#include "TMath.h"
#include "TH3.h"

#include "TGLPlotCamera.h"
#include "TGL5DPainter.h"
#include "TGL5D.h"

ClassImp(TGL5DDataSet);

/** \class TGL5DDataSet
\ingroup opengl
*/

namespace {

void FindRange(Long64_t size, const Double_t *src, Rgl::Range_t &range);

}

////////////////////////////////////////////////////////////////////////////////
///Constructor. Reads data from TTree,
///estimates ranges, creates a painter.

TGL5DDataSet::TGL5DDataSet(TTree *tree)
               : TNamed("TGL5DataSet", "TGL5DataSet"),
                 fNP(0),
                 fV1(0), fV2(0), fV3(0), fV4(0), fV5(0),
                 fV1Range(1.), fV2Range(1.), fV3Range(1.),
                 fV4IsString(kFALSE)
{
   if (!tree) {
      Error("TGL5Data", "Null pointer tree.");
      throw std::runtime_error("");
   }

   fNP = tree->GetSelectedRows();

   Info("TGL5DDataSet", "Number of selected rows: %d", Int_t(fNP))   ;
   //Now, let's access the data and find ranges.
   fV1 = tree->GetVal(0);
   fV2 = tree->GetVal(1);
   fV3 = tree->GetVal(2);
   fV4 = tree->GetVal(3);
   fV5 = tree->GetVal(4);
   //
   fV4IsString = tree->GetVar(3)->IsString();
   //
   if (!fV1 || !fV2 || !fV3 || !fV4 || !fV5) {
      Error("TGL5DDataSet", "One or all of vN is a null pointer.");
      throw std::runtime_error("");
   }
   //
   FindRange(fNP, fV1, fV1MinMax);
   FindRange(fNP, fV2, fV2MinMax);
   FindRange(fNP, fV3, fV3MinMax);
   FindRange(fNP, fV4, fV4MinMax);
   FindRange(fNP, fV5, fV5MinMax);
   //
   const Double_t v1Add = 0.1 * (fV1MinMax.second - fV1MinMax.first);
   const Double_t v2Add = 0.1 * (fV2MinMax.second - fV2MinMax.first);
   const Double_t v3Add = 0.1 * (fV3MinMax.second - fV3MinMax.first);
   //Adjust ranges.
   fV1MinMax.first  -= v1Add, fV1MinMax.second += v1Add;
   fV1Range = fV1MinMax.second - fV1MinMax.first;
   fV2MinMax.first  -= v2Add, fV2MinMax.second += v2Add;
   fV2Range = fV2MinMax.second - fV2MinMax.first;
   fV3MinMax.first  -= v3Add, fV3MinMax.second += v3Add;
   fV3Range = fV3MinMax.second - fV3MinMax.first;
   //Set axes.
   TH3F hist("tmp", "tmp", 2, -1., 1., 2, -1., 1., 2, -1., 1.);
   //TAxis has a lot of attributes, defaults, set by ctor,
   //are not enough to be correctly painted by TGaxis object.
   //To simplify their initialization - I use temporary histogram.
   hist.GetXaxis()->Copy(fXAxis);
   hist.GetYaxis()->Copy(fYAxis);
   hist.GetZaxis()->Copy(fZAxis);

   fXAxis.Set(kDefaultNB, fV1MinMax.first, fV1MinMax.second);
   fYAxis.Set(kDefaultNB, fV2MinMax.first, fV2MinMax.second);
   fZAxis.Set(kDefaultNB, fV3MinMax.first, fV3MinMax.second);

   fPainter.reset(new TGLHistPainter(this));
   SetBit(kCanDelete);//TPad will delete this object when closed.
}

////////////////////////////////////////////////////////////////////////////////
///Check, if the object is under cursor.

Int_t TGL5DDataSet::DistancetoPrimitive(Int_t px, Int_t py)
{
   return fPainter->DistancetoPrimitive(px, py);
}

////////////////////////////////////////////////////////////////////////////////
///Action.

void TGL5DDataSet::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   return fPainter->ExecuteEvent(event, px, py);
}

////////////////////////////////////////////////////////////////////////////////
///Info for status bar.

char *TGL5DDataSet::GetObjectInfo(Int_t /*px*/, Int_t /*py*/) const
{
   static char mess[] = {"5d data set"};
   return mess;
}

////////////////////////////////////////////////////////////////////////////////
///Paint.

void TGL5DDataSet::Paint(Option_t * /*option*/)
{
   fPainter->Paint("dummyoption");
}

////////////////////////////////////////////////////////////////////////////////
///Get access to painter (for GUI-editor).

TGL5DPainter *TGL5DDataSet::GetRealPainter()const
{
   return static_cast<TGL5DPainter *>(fPainter->GetRealPainter());
}

////////////////////////////////////////////////////////////////////////////////
///"Select" sub-range from source data
///- remember indices of "good" points.

void TGL5DDataSet::SelectPoints(Double_t v4Level, Double_t range)
{
   fIndices.clear();

   for (Int_t i = 0; i < fNP; ++i)
      if (TMath::Abs(fV4[i] - v4Level) < range)
         fIndices.push_back(i);
}

////////////////////////////////////////////////////////////////////////////////
///Size of selected sub-range.

UInt_t TGL5DDataSet::SelectedSize()const
{
   return UInt_t(fIndices.size());
}

////////////////////////////////////////////////////////////////////////////////
///V1 from sub-range, converted to unit cube.

Double_t TGL5DDataSet::V1(UInt_t ind)const
{
   return V1ToUnitCube(fV1[fIndices[ind]]);
}

////////////////////////////////////////////////////////////////////////////////
///V2 from sub-range, converted to unit cube.

Double_t TGL5DDataSet::V2(UInt_t ind)const
{
   return V2ToUnitCube(fV2[fIndices[ind]]);
}

////////////////////////////////////////////////////////////////////////////////
///V3 from sub-range, converted to unit cube.

Double_t TGL5DDataSet::V3(UInt_t ind)const
{
   return V3ToUnitCube(fV3[fIndices[ind]]);
}

////////////////////////////////////////////////////////////////////////////////
///X axis for plot.

TAxis *TGL5DDataSet::GetXAxis()const
{
   return &fXAxis;
}

////////////////////////////////////////////////////////////////////////////////
///Y axis for plot.

TAxis *TGL5DDataSet::GetYAxis()const
{
   return &fYAxis;
}

////////////////////////////////////////////////////////////////////////////////
///Z axis for plot.

TAxis *TGL5DDataSet::GetZAxis()const
{
   return &fZAxis;
}

////////////////////////////////////////////////////////////////////////////////
///V1 range (X).

const Rgl::Range_t &TGL5DDataSet::GetXRange()const
{
   return fV1MinMax;
}

////////////////////////////////////////////////////////////////////////////////
///V2 range (Y).

const Rgl::Range_t &TGL5DDataSet::GetYRange()const
{
   return fV2MinMax;
}

////////////////////////////////////////////////////////////////////////////////
///V3 range (Z).

const Rgl::Range_t &TGL5DDataSet::GetZRange()const
{
   return fV3MinMax;
}

////////////////////////////////////////////////////////////////////////////////
///V4 range.

const Rgl::Range_t &TGL5DDataSet::GetV4Range()const
{
   return fV4MinMax;
}

////////////////////////////////////////////////////////////////////////////////
///V1 to unit cube.

Double_t TGL5DDataSet::V1ToUnitCube(Double_t v1)const
{
   return (v1 - fV1MinMax.first) / fV1Range;
}

////////////////////////////////////////////////////////////////////////////////
///V2 to unit cube.

Double_t TGL5DDataSet::V2ToUnitCube(Double_t v2)const
{
   return (v2 - fV2MinMax.first) / fV2Range;
}

////////////////////////////////////////////////////////////////////////////////
///V3 to unit cube.

Double_t TGL5DDataSet::V3ToUnitCube(Double_t v3)const
{
   return (v3 - fV3MinMax.first) / fV3Range;
}

namespace {

////////////////////////////////////////////////////////////////////////////////
///Find both min and max on a range in one pass through sequence.

void FindRange(Long64_t size, const Double_t *src, Rgl::Range_t &range)
{
   range.first  = src[0];
   range.second = src[0];

   for (Long64_t i = 1; i < size; ++i) {
      range.first  = TMath::Min(range.first,  src[i]);
      range.second = TMath::Max(range.second, src[i]);
   }
}

}
