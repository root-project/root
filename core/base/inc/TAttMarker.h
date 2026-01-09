// @(#)root/base:$Id$
// Author: Rene Brun   12/05/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAttMarker
#define ROOT_TAttMarker


#include "Rtypes.h"


class TAttMarker {

protected:
   Color_t    fMarkerColor;       ///< Marker color index
   Style_t    fMarkerStyle;       ///< Marker style
   Size_t     fMarkerSize;        ///< Marker size

public:
   TAttMarker();
   TAttMarker(Color_t markerColor, Style_t style, Size_t msize);
   virtual ~TAttMarker();
          void     Copy(TAttMarker &attmarker) const;
   virtual Color_t  GetMarkerColor() const {return fMarkerColor;} ///< Return the marker color
   virtual Style_t  GetMarkerStyle() const {return fMarkerStyle;} ///< Return the marker style
   virtual Size_t   GetMarkerSize()  const {return fMarkerSize;}  ///< Return the marker size
   virtual void     Modify();
   virtual void     ResetAttMarker(Option_t *option="");
   virtual void     SaveMarkerAttributes(std::ostream &out, const char *name, Int_t coldef=1, Int_t stydef=1, Int_t sizdef=1);
   virtual void     SetMarkerAttributes();  // *MENU*
   virtual void     SetMarkerColor(Color_t tmarkerColor); ///< Set the marker color
   virtual void     SetMarkerColorAlpha(Color_t tmarkerColor, Float_t malpha);
   virtual void     SetMarkerStyle(Style_t style); ///< Set the marker style
   virtual void     SetMarkerSize(Size_t msize);

   ClassDef(TAttMarker,2)  //Marker attributes
};

enum EMarkerStyle {kDot=1, kPlus, kStar, kCircle=4, kMultiply=5,
                   kFullDotSmall=6, kFullDotMedium=7, kFullDotLarge=8,
                   kFullCircle=20, kFullSquare=21, kFullTriangleUp=22,
                   kFullTriangleDown=23, kOpenCircle=24, kOpenSquare=25,
                   kOpenTriangleUp=26, kOpenDiamond=27, kOpenCross=28,
                   kOpenStar=29, kFullStar=30, kOpenTriangleDown=32,
                   kFullDiamond=33, kFullCross=34, kOpenDiamondCross=41,
                   kOpenCrossX=43, kFullThreeTriangles=45,
                   kOpenThreeTriangles=47, kFourSquaresX=48,
                   kFourSquaresPlus=49 };

#endif