// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov, 14/8/2011

/*************************************************************************
 * Copyright (C) 1995-2011, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_IOSPainter
#define ROOT_IOSPainter

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// IOSPainter                                                           //
//                                                                      //
// Graphics operations required by IOSPad are implemented in            //
// IOSPainter. Uses CoreGraphics and CoreText.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>

#include <CoreGraphics/CoreGraphics.h>


#ifndef ROOT_TVirtualPadPainter
#include "TVirtualPadPainter.h"
#endif
#ifndef ROOT_TextOperations
#include "IOSTextOperations.h"
#endif
#ifndef ROOT_GraphicUtils
#include "IOSGraphicUtils.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

namespace ROOT {
namespace iOS {

//
//SpaceConverter converts coordinates from pad's user space into UIView's userspace.
//

class SpaceConverter {
public:
   SpaceConverter();
   SpaceConverter(UInt_t viewW, Double_t xMin, Double_t xMax,
                  UInt_t viewH, Double_t yMin, Double_t yMax);
   
   void SetConverter(UInt_t viewW, Double_t xMin, Double_t xMax, 
                     UInt_t viewH, Double_t yMin, Double_t yMax);
   
   Double_t XToView(Double_t x)const;
   Double_t YToView(Double_t y)const;
   
private:
   Double_t fXMin;
   Double_t fXConv;
   
   Double_t fYMin;
   Double_t fYConv;
};

class Painter : public TVirtualPadPainter {
public:

   Painter();
 
   Bool_t   IsTransparent() const {return fRootOpacity < 100;}//+
   void     SetOpacity(Int_t percent);
   
   //Now, drawing primitives.
   void     DrawLine(Double_t x1, Double_t y1, Double_t x2, Double_t y2);
   void     DrawLineNDC(Double_t u1, Double_t v1, Double_t u2, Double_t v2);
   
   void     DrawBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2, EBoxMode mode);
   
   void     DrawFillArea(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawFillArea(Int_t n, const Float_t *x, const Float_t *y);
      
   void     DrawPolyLine(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawPolyLine(Int_t n, const Float_t *x, const Float_t *y);
   void     DrawPolyLineNDC(Int_t n, const Double_t *u, const Double_t *v);
   
   void     DrawPolyMarker(Int_t n, const Double_t *x, const Double_t *y);
   void     DrawPolyMarker(Int_t n, const Float_t *x, const Float_t *y);
   
   void     DrawText(Double_t x, Double_t y, const char *text, ETextMode mode);
   void     DrawTextNDC(Double_t u, Double_t v, const char *text, ETextMode mode);
   
   void     SetContext(CGContextRef ctx);
   void     SetTransform(UInt_t w, Double_t xMin, Double_t xMax, UInt_t h, Double_t yMin, Double_t yMax);
   
   Int_t    CreateDrawable(UInt_t, UInt_t){return 0;}
   void     ClearDrawable(){}
   void     CopyDrawable(Int_t, Int_t, Int_t){}
   void     DestroyDrawable(){}
   void     SelectDrawable(Int_t) {}
   
   void     SaveImage(TVirtualPad *, const char *, Int_t) const{}

   enum EMode {
      kPaintToSelectionBuffer, //A pad draws the scene into the selection buffer.
      kPaintToView,            //Normal painting (normal colors and styles).
      kPaintShadow,            //Paint the gray polygon/line under selected object (shadow).
      kPaintSelected,          //Only selected object is painted (special style and color).
      kPaintThumbnail          //Paint into small picture, very low level of details.
   };

   //Temporary solution for objecti picking in a pad.
   void     SetPainterMode(EMode mode)
   {
      fPainterMode = mode;
   }
   
   void     SetCurrentObjectID(UInt_t objId)
   {
      fCurrentObjectID = objId;
   }
   
   void GetTextExtent(UInt_t &w, UInt_t &h, const char *text);
   
   //Line attributes to be set up in TPad.
   Color_t  GetLineColor() const;
   Style_t  GetLineStyle() const;
   Width_t  GetLineWidth() const;
   
   void     SetLineColor(Color_t lcolor);
   void     SetLineStyle(Style_t lstyle);
   void     SetLineWidth(Width_t lwidth);
   
   //Fill attributes to be set up in TPad.
   Color_t  GetFillColor() const;
   Style_t  GetFillStyle() const;

   void     SetFillColor(Color_t fcolor);
   void     SetFillStyle(Style_t fstyle);
   
   //Text attributes.
   Short_t  GetTextAlign() const;
   Float_t  GetTextAngle() const;
   Color_t  GetTextColor() const;
   Font_t   GetTextFont() const;
   Float_t  GetTextSize() const;
   Float_t  GetTextMagnitude() const;
   
   void     SetTextAlign(Short_t align);
   void     SetTextAngle(Float_t tangle);
   void     SetTextColor(Color_t tcolor);
   void     SetTextFont(Font_t tfont);
   void     SetTextSize(Float_t tsize);
   void     SetTextSizePixels(Int_t);
   
private:

   //Polygon parameters.
   //Polygon stipple here.
   void     SetStrokeParameters()const;
   void     SetPolygonParameters()const;
   void     SetMarkerColor()const;
   Bool_t   PolygonHasStipple()const;
   
   //
   void     FillBoxWithPattern(Double_t x1, Double_t y1, Double_t x2, Double_t y2)const;
   void     FillBox(Double_t x1, Double_t y1, Double_t x2, Double_t y2)const;
   void     DrawBoxOutline(Double_t x1, Double_t y1, Double_t x2, Double_t y2)const;

   void     FillAreaWithPattern(Int_t n, const Double_t *x, const Double_t *y)const;   
   void     FillArea(Int_t n, const Double_t *x, const Double_t *y)const;
   
   void     DrawText(Double_t x, Double_t y, const CTLineGuard &ctLine);
   
   //
   FontManager     fFontManager;
   CGContextRef    fCtx;//Quartz context.
   SpaceConverter  fConverter;   

   Int_t           fRootOpacity;

   
   typedef std::vector<TPoint>::size_type size_type;
   std::vector<TPoint>    fPolyMarker;//Buffer for converted poly-marker coordinates.

   //Staff for picking.
   EMode fPainterMode;
   UInt_t fCurrentObjectID;
   GraphicUtils::IDEncoder fEncoder;
   
   void SetLineColorForCurrentObjectID() const;
   void SetPolygonColorForCurrentObjectID() const;
   void SetLineColorHighlighted() const;

   Painter(const Painter &rhs);
   Painter &operator = (const Painter &rhs);
};

}//iOS
}//namespace ROOT

#endif
