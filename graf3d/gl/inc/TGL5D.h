// @(#)root/gl:$Id$
// Author: Timur Pocheptsov  2009
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGL5D
#define ROOT_TGL5D

#include <memory>
#include <vector>

#ifndef ROOT_TGLHistPainter
#include "TGLHistPainter.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TAxis
#include "TAxis.h"
#endif

class TGL5DPainter;
class TTree;

//TGL5D is a class to setup TGL5DPainter from TTree,
//hold data pointers, select required ranges,
//convert them into unit cube.
class TGL5DDataSet : public TNamed {
   friend class TGL5DPainter;
private:
   enum Edefaults{
      kDefaultNB = 50//Default number of bins along X,Y,Z axes.
   };
public:
   TGL5DDataSet(TTree *inputData);

   //These are functions for TPad and
   //TPad's standard machinery (picking, painting).
   Int_t    DistancetoPrimitive(Int_t px, Int_t py);
   void     ExecuteEvent(Int_t event, Int_t px, Int_t py);
   char    *GetObjectInfo(Int_t px, Int_t py) const;
   void     Paint(Option_t *option);

   //This is for editor.
   TGL5DPainter *GetRealPainter()const;

   //Select points for iso-surface.
   void     SelectPoints(Double_t v4Level, Double_t range);
   UInt_t   SelectedSize()const;

   //Take a point from selected sub-range (V1 == X, V2 == Y, V3 == Z for 3D).
   Double_t V1(UInt_t ind)const;
   Double_t V2(UInt_t ind)const;
   Double_t V3(UInt_t ind)const;

   //Very similar to TH3's axes.
   TAxis   *GetXAxis()const;
   TAxis   *GetYAxis()const;
   TAxis   *GetZAxis()const;

   //Data ranges for V1, V2, V3, V4.
   const Rgl::Range_t &GetXRange()const;
   const Rgl::Range_t &GetYRange()const;
   const Rgl::Range_t &GetZRange()const;
   const Rgl::Range_t &GetV4Range()const;

private:
   //These three functions for TKDEFGT,
   //which will convert all point coordinates
   //into unit cube before density estimation.
   Double_t V1ToUnitCube(Double_t v1)const;
   Double_t V2ToUnitCube(Double_t v2)const;
   Double_t V3ToUnitCube(Double_t v3)const;

   Long64_t        fNP;//Number of entries.
   const Double_t *fV1;//V1.
   const Double_t *fV2;//V2.
   const Double_t *fV3;//V3.
   const Double_t *fV4;//V4.
   const Double_t *fV5;//V5.

   //These are fixed ranges of the data set,
   //calculated during construction.
   Rgl::Range_t    fV1MinMax;//V1 range.
   Double_t        fV1Range;//max - min.
   Rgl::Range_t    fV2MinMax;//V2 range.
   Double_t        fV2Range;//max - min.
   Rgl::Range_t    fV3MinMax;//V3 range.
   Double_t        fV3Range;//max - min.
   Rgl::Range_t    fV4MinMax;//V4 range.
   Rgl::Range_t    fV5MinMax;//V5 range.

   //This are ranges and bin numbers
   //for plot, inside fixed ranges.
   mutable TAxis   fXAxis;
   mutable TAxis   fYAxis;
   mutable TAxis   fZAxis;
   //V4 can have a string type.
   Bool_t          fV4IsString;
   //Painter to visualize dataset.
   std::auto_ptr<TGLHistPainter> fPainter;
   //Indices of points, selected for some iso-level.
   std::vector<UInt_t> fIndices;

   TGL5DDataSet(const TGL5DDataSet &rhs);
   TGL5DDataSet &operator = (const TGL5DDataSet &rhs);

   ClassDef(TGL5DDataSet, 0)//Class to read data from TTree and create TGL5DPainter.
};

#endif
