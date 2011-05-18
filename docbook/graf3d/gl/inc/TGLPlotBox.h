// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  31/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPlotFrame
#define ROOT_TGLPlotFrame

#include <vector>

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

class TColor;

/*
   TGLPlotBox draws a box behind a plot.
*/

class TGLPlotBox {
private:
   const TColor          *fFrameColor;
   const Bool_t           fXOYSelectable;
   const Bool_t           fXOZSelectable;
   const Bool_t           fYOZSelectable;

   Bool_t                 fSelectablePairs[4][2];

   TGLVertex3             f3DBox[8];
   mutable TGLVertex3     f2DBox[8];
   mutable TGLVertex3     f2DBoxU[8];
   mutable Int_t          fFrontPoint;

   //The sizes of a "unit" cube.
   Double_t               fRangeXU;
   Double_t               fRangeYU;
   Double_t               fRangeZU;

public:

   TGLPlotBox(Bool_t xoySelectable, Bool_t xozSelectable, Bool_t yozSelectable);
   //ClassDef macro adds some virtual functions,
   //so, to supress g++ warnings virtual destructor declared.
   virtual ~TGLPlotBox();

   void DrawBox(Int_t selectedPart, Bool_t selectionPass,
                const std::vector<Double_t> &zLevels,
                Bool_t highColor)const;

   void SetPlotBox(const Rgl::Range_t &xRange,
                   const Rgl::Range_t &yRange,
                   const Rgl::Range_t &zRange);
   void SetPlotBox(const Rgl::Range_t &xRange, Double_t rangeXU,
                   const Rgl::Range_t &yRange, Double_t rangeYU,
                   const Rgl::Range_t &zRange, Double_t rangeZU);

   void SetFrameColor(const TColor *color);

   Int_t FindFrontPoint()const;
   Int_t GetFrontPoint()const;

   const TGLVertex3 *Get3DBox()const;
   const TGLVertex3 *Get2DBox()const;

   static const Int_t    fgFramePlanes[][4];
   static const Int_t    fgBackPairs[][2];
   static const Double_t fgNormals[][3];

private:
   void DrawBackPlane(Int_t plane, Bool_t selectionPass,
                      const std::vector<Double_t> &zLevels)const;

   ClassDef(TGLPlotBox, 0)//Back box for plot.
};

#endif
