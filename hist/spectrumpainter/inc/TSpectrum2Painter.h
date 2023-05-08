// @(#)root/spectrumpainter:$Id: TSpectrum2Painter.h,v 1.0
// Author: Miroslav Morhac 29/09/06

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSpectrum2Painter
#define ROOT_TSpectrum2Painter


#include "TNamed.h"

class TH2;
class TLine;
class TColor;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSpectrum2Painter Algorithms                                         //
//                                                                      //
// 3D graphics representations package.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TSpectrum2Painter: public TNamed {

public:
   TSpectrum2Painter(TH2* h2, Int_t bs);
   ~TSpectrum2Painter() override;

   void GetAngles(Int_t &alpha,Int_t &beta,Int_t &view);
   void GetBezier(Int_t &bezier);
   void GetChanGrid(Int_t &enable,Int_t &color);
   void GetChanMarks(Int_t &enable,Int_t &color,Int_t &width,Int_t &height,Int_t &style);
   void GetColorAlgorithm(Int_t &colorAlgorithm);
   void GetColorIncrements(Double_t &r,Double_t &g,Double_t &b);
   void GetContourWidth(Int_t &width);
   void GetDisplayMode(Int_t &modeGroup,Int_t &displayMode);
   void GetLightHeightWeight(Double_t &weight);
   void GetLightPosition(Int_t &x,Int_t &y,Int_t &z);
   void GetNodes(Int_t &nodesx,Int_t &nodesy);
   void GetPenAttr(Int_t &color, Int_t &style, Int_t &width);
   void GetShading(Int_t &shading,Int_t &shadow);
   void GetZScale(Int_t &scale);
   void Paint(Option_t *option) override;
   void SetAngles(Int_t alpha,Int_t beta,Int_t view);
   void SetBezier(Int_t bezier);
   void SetChanGrid(Int_t enable,Int_t color);
   void SetChanMarks(Int_t enable,Int_t color,Int_t width,Int_t height,Int_t style);
   void SetColorAlgorithm(Int_t colorAlgorithm);
   void SetColorIncrements(Double_t r,Double_t g,Double_t b);
   void SetContourWidth(Int_t width);
   void SetDisplayMode(Int_t modeGroup,Int_t displayMode);
   void SetLightHeightWeight(Double_t weight);
   void SetLightPosition(Int_t x,Int_t y,Int_t z);
   void SetNodes(Int_t nodesx,Int_t nodesy);
   void SetPenAttr(Int_t color,Int_t style,Int_t width);
   void SetShading(Int_t shading,Int_t shadow);
   void SetZScale(Int_t scale);

   static void PaintSpectrum(TH2* h2, Option_t *option="",Int_t bs=1600);

   enum {
      kModeGroupSimple=0,
      kModeGroupHeight=1,
      kModeGroupLight=2,
      kModeGroupLightHeight=3,
      kDisplayModePoints=1,
      kDisplayModeGrid=2,
      kDisplayModeContours=3,
      kDisplayModeBars=4,
      kDisplayModeLinesX=5,
      kDisplayModeLinesY=6,
      kDisplayModeBarsX=7,
      kDisplayModeBarsY=8,
      kDisplayModeNeedles=9,
      kDisplayModeSurface=10,
      kDisplayModeTriangles=11,
      kZScaleLinear=0,
      kZScaleLog=1,
      kZScaleSqrt=2,
      kColorAlgRgbSmooth=0,
      kColorAlgRgbModulo=1,
      kColorAlgCmySmooth=2,
      kColorAlgCmyModulo=3,
      kColorAlgCieSmooth=4,
      kColorAlgCieModulo=5,
      kColorAlgYiqSmooth=6,
      kColorAlgYiqModulo=7,
      kColorAlgHvsSmooth=8,
      kColorAlgHvsModulo=9,
      kShadowsNotPainted=0,
      kShadowsPainted=1,
      kNotShaded=0,
      kShaded=1,
      kNoBezierInterpol=0,
      kBezierInterpol=1,
      kPenStyleSolid=1,
      kPenStyleDash=2,
      kPenStyleDot=3,
      kPenStyleDashDot=4,
      kChannelMarksNotDrawn=0,
      kChannelMarksDrawn=1,
      kChannelMarksStyleDot=1,
      kChannelMarksStyleCross=2,
      kChannelMarksStyleStar=3,
      kChannelMarksStyleRectangle=4,
      kChannelMarksStyleX=5,
      kChannelMarksStyleDiamond=6,
      kChannelMarksStyleTriangle=7,
      kChannelGridNotDrawn=0,
      kChannelGridDrawn=1
  };

protected:
   TH2      *fH2;            //pointer to 2D histogram TH2
   Int_t     fXmin;          //x-starting channel of spectrum
   Int_t     fXmax;          //x-end channel of spectrum
   Int_t     fYmin;          //y-starting channel of spectrum
   Int_t     fYmax;          //y-end channel of spectrum
   Double_t  fZmin;          //base counts
   Double_t  fZmax;          //counts full scale
   Int_t     fBx1;           //positon of picture on Canvas, min x
   Int_t     fBx2;           //positon of picture on Canvas, max x
   Int_t     fBy1;           //positon of picture on Canvas, min y
   Int_t     fBy2;           //positon of picture on Canvas, max y
   Int_t     fPenColor;      //color of spectrum
   Int_t     fPenDash;       //style of pen
   Int_t     fPenWidth;      //width of line
   Int_t     fModeGroup;     //display mode algorithm group (simple modes-kModeGroupSimple, modes with shading according to light-kModeGroupLight, modes with shading according to channels counts-kModeGroupHeight, modes of combination of shading according to light and to channels counts-kModeGroupLightHeight)
   Int_t     fDisplayMode;   //spectrum display mode (points, grid, contours, bars, x_lines, y_lines, bars_x, bars_y, needles, surface, triangles)
   Int_t     fZscale;        //z scale (linear, log, sqrt)
   Int_t     fNodesx;        //number of nodes in x dimension of grid
   Int_t     fNodesy;        //number of nodes in y dimension of grid
   Int_t     fContWidth;     //width between contours, applies only for contours display mode
   Int_t     fAlpha;         //angles of display,alfa+beta must be less or equal to 90, alpha- angle between base line of Canvas and right lower edge of picture base plane
   Int_t     fBeta;          //angle between base line of Canvas and left lower edge of picture base plane
   Int_t     fViewAngle;     //rotation angle of the view, it can be 0, 90, 180, 270 degrees
   Int_t     fLevels;        //# of color levels for rainbowed display modes, it does not apply for simple display modes algorithm group
   Double_t  fRainbow1Step;  //determines the first component  step for neighbouring color levels, applies only for rainbowed display modes, it does not apply for simple display modes algorithm group
   Double_t  fRainbow2Step;  //determines the second component  step for neighbouring color levels, applies only for rainbowed display modes, it does not apply for simple display modes algorithm group
   Double_t  fRainbow3Step;  //determines the third component  step for neighbouring color levels, applies only for rainbowed display modes, it does not apply for simple display modes algorithm group
   Int_t     fColorAlg;      //applies only for rainbowed display modes (rgb smooth alorithm, rgb modulo color component, cmy smooth alorithm, cmy modulo color component, cie smooth alorithm, cie modulo color component, yiq smooth alorithm, yiq modulo color component, hsv smooth alorithm, hsv modulo color component, it does not apply for simple display modes algorithm group
   Double_t  fLHweight;      //weight between shading according to fictive light source and according to channels counts, applies only for kModeGroupLightHeight modes group
   Int_t     fXlight;        //x position of fictive light source, applies only for rainbowed display modes with shading according to light
   Int_t     fYlight;        //y position of fictive light source, applies only for rainbowed display modes with shading according to light
   Int_t     fZlight;        //z position of fictive light source, applies only for rainbowed display modes with shading according to light
   Int_t     fShadow;        //determines whether shadow will be drawn (no shadow, shadow), for rainbowed display modes with shading according to light
   Int_t     fShading;       //determines whether the picture will shaded, smoothed (no shading, shading), for rainbowed display modes only
   Int_t     fBezier;        //determines Bezier interpolation (applies only for simple display modes group for grid, x_lines, y_lines display modes)
   Int_t     fChanmarkEnDis; //decides whether the channel marks are shown
   Int_t     fChanmarkStyle; //style of channel marks
   Int_t     fChanmarkWidth; //width of channel marks
   Int_t     fChanmarkHeight;//height of channel marks
   Int_t     fChanmarkColor; //color of channel marks
   Int_t     fChanlineEnDis; //decides whether the channel lines (grid) are shown
   Int_t     fChanlineColor; //color of channel lines (grid)

   //auxiliary variables,transformation coeffitients for internal use only
   Double_t  fKx;
   Double_t  fKy;
   Double_t  fMxx;
   Double_t  fMxy;
   Double_t  fMyx;
   Double_t  fMyy;
   Double_t  fTxx;
   Double_t  fTxy;
   Double_t  fTyx;
   Double_t  fTyy;
   Double_t  fTyz;
   Double_t  fVx;
   Double_t  fVy;
   Double_t  fNuSli;

   //auxiliary internal variables, working place
   Double_t  fZ,fZeq,fGbezx,fGbezy,fDxspline,fDyspline,fZPresetValue;
   Int_t     fXt,fYt,fXs,fYs,fXe,fYe,fLine;
   Short_t  *fEnvelope;                 //!
   Short_t  *fEnvelopeContour;          //!
   TColor   *fNewColor;                 //!
   Int_t     fMaximumXScreenResolution; //!buffers' size
   Int_t     fNewColorIndex;
   Int_t     fBzX[4];
   Int_t     fBzY[4];

   Int_t    BezC(Int_t i);
   Double_t BezierBlend(Int_t i,Double_t bezf);
   void     BezierSmoothing(Double_t bezf);
   Double_t ColorCalculation(Double_t dx1,Double_t dy1,Double_t z1,Double_t dx2,Double_t dy2,Double_t z2,Double_t dx3,Double_t dy3,Double_t z3);//calculation of new color
   void     ColorModel(unsigned ui,unsigned ui1,unsigned ui2,unsigned ui3);//calculation of color according to chosen algorithm
   void     CopyEnvelope(Double_t xr,Double_t xs,Double_t yr,Double_t ys);
   void     DrawMarker(Int_t x,Int_t y,Int_t w,Int_t h,Int_t type);
   void     Envelope(Int_t x1,Int_t y1,Int_t x2,Int_t y2);
   void     EnvelopeBars(Int_t x1,Int_t y1,Int_t x2,Int_t y2);
   Double_t ShadowColorCalculation(Double_t xtaz,Double_t ytaz,Double_t ztaz,Double_t shad_noise);//shadow color
   void     Slice(Double_t xr,Double_t yr,Double_t xs,Double_t ys,TLine *line);
   void     Transform(Int_t it,Int_t jt,Int_t zmt);//transform function

public:
   ClassDefOverride(TSpectrum2Painter,0)   //TSpectrum 3d graphics package

private:
   TSpectrum2Painter (const TSpectrum2Painter&);
   TSpectrum2Painter& operator=(const TSpectrum2Painter&);

};

#endif
