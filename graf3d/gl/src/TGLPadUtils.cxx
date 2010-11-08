// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include <stdexcept>

#include "TVirtualPad.h"
#include "TVirtualX.h"
#include "RStipples.h"
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"

#include "TGLPadUtils.h"
#include "TGLIncludes.h"

namespace Rgl {
namespace Pad {

const UInt_t PolygonStippleSet::fgBitSwap[] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};


/*
Temporary fix.
*/
#ifndef GL_VERSION_1_2
const GLenum lineWidthPNAME = GLenum(0xB22);
const GLenum pointSizePNAME = GLenum(0xB12);
#else
const GLenum lineWidthPNAME = GLenum(GL_SMOOTH_LINE_WIDTH_RANGE);//Cast for real enums and macros.
const GLenum pointSizePNAME = GLenum(GL_SMOOTH_POINT_SIZE_RANGE);
#endif

/*
Auxiliary class to converts ROOT's polygon stipples from
RStipples.h into GL's stipples and hold them in a fStipples array.
*/
//______________________________________________________________________________
PolygonStippleSet::PolygonStippleSet()
{
    /*
    I have to assume, that gStipple has two chars in a line.
    There in no way to calculate line length and there are no corresponding constants in RStipple.h.
    So, these numbers are hardcode here.
    Ordering in RStipples completely different from OpenGL. 
    In OpenGL, if I have, say, 16x2 pattern, GLbytes will be:
    
    [3][4]
    [1][2]
    
    and bits inside them
    
    [7 6 5 4 3 2 1 0][7 6 5 4 3 2 1 0]
    [7 6 5 4 3 2 1 0][7 6 5 4 3 2 1 0].
    
    But for X11 this will be:
    
    [2][1]
    [4][3]
    
    [0 1 2 3 4 5 6 7][0 1 2 3 4 5 6 7]
    [0 1 2 3 4 5 6 7][0 1 2 3 4 5 6 7]
    
    So, line 0x7, 0xE from X11 must be 
    converted into 0x70, 0xE0 for OpenGL.
    
    As OpenGL expects 32x32 pattern, I have to twice each line.
   */
   
   /*If somebody will seriously change gStipples declaration, 
   so, that sizeof gStipples becomes "wrong", change this!*/
   const UInt_t numOfStipples = sizeof gStipples / sizeof gStipples[0];
   fStipples.resize(kStippleSize * numOfStipples);

   for (UInt_t i = 0; i < numOfStipples; ++i) {
      const UInt_t baseInd = i * kStippleSize;
      
      for (Int_t j = 15, j1 = 0; j >= 0; --j, ++j1) {//ROOT uses 16x16 stipples.
         const UInt_t rowShift = j1 * kRowSize;
         
         for (Int_t k = 1, k1 = 0; k >= 0; --k, ++k1) {//Two chars form a line.
            const UChar_t pixel = SwapBits(gStipples[i][j * 2 + k]);
            const UInt_t ind = baseInd + rowShift + k1;
            
            fStipples[ind]      = pixel;
            fStipples[ind + 2]  = pixel;
            fStipples[ind + 64] = pixel;
            fStipples[ind + 66] = pixel;
         }
      }
   }
}

//______________________________________________________________________________
UInt_t PolygonStippleSet::SwapBits(UInt_t b)
{
   b &= k16Bits;
   
   const UInt_t low = fgBitSwap[b & kLow4] << 4;
   const UInt_t up  = fgBitSwap[(b & kUp4) >> 4];
   
   return low | up;
}

/*
Class to manipulate fill parameters.
*/
//______________________________________________________________________________
FillAttribSet::FillAttribSet(const PolygonStippleSet &set, Bool_t ignoreStipple)
                  : fStipple(0)
{
   //Polygon stipple, if required.
   const UInt_t style = gVirtualX->GetFillStyle() / 1000;
   
   if (!ignoreStipple) {
      if (style == 3) {
         const UInt_t fasi  = gVirtualX->GetFillStyle() % 1000;
         fStipple = (fasi >= 1 && fasi <=25) ? fasi : 2;
         glPolygonStipple(&set.fStipples[fStipple * PolygonStippleSet::kStippleSize]);
         glEnable(GL_POLYGON_STIPPLE);
      }
   }
   
   //Color.
   Float_t rgba[] = {0.f, 0.f, 0.f, 1.f};
   ExtractRGB(gVirtualX->GetFillColor(), rgba);
   glColor4fv(rgba);
}

//______________________________________________________________________________
FillAttribSet::~FillAttribSet()
{
   if (fStipple)
      glDisable(GL_POLYGON_STIPPLE);
}

/*
"ROOT like" line stipples.
*/

const UShort_t gLineStipples[] = {0xffff, 0xffff, 0x3333, 0x5555, 
                                  0xf040, 0xf4f4, 0xf111, 0xf0f0, 
                                  0xff11, 0x3fff, 0x08ff};

const UInt_t gMaxStipple = sizeof gLineStipples / sizeof gLineStipples[0];

/*
Set/unset line attributes.
*/
//______________________________________________________________________________
LineAttribSet::LineAttribSet(Bool_t smooth, UInt_t stipple, Double_t maxWidth, Bool_t setWidth)
                  : fSmooth(smooth), fStipple(stipple), fSetWidth(setWidth)
{
   //Set up line parameters.
   //Smooth.
   if (fSmooth) {
      glEnable(GL_BLEND);
      glEnable(GL_LINE_SMOOTH);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
   }
   
   //Stipple.
   if (fStipple > 1) {
      if (fStipple >= gMaxStipple)
         fStipple = 1;
      else {
         glEnable(GL_LINE_STIPPLE);
         glLineStipple(fStipple == 10 ? 2 : 1, gLineStipples[fStipple]);
      }
   }
   
   //Color.
   Float_t rgba[] = {0.f, 0.f, 0.f, 0.8f};
   ExtractRGB(gVirtualX->GetLineColor(), rgba);
   glColor4fv(rgba);
   //Width.
   if (fSetWidth) {
      const Width_t w = gVirtualX->GetLineWidth();
      glLineWidth(w > maxWidth ? maxWidth : !w ? 1.f : w);
   }
}

//______________________________________________________________________________
LineAttribSet::~LineAttribSet()
{
   if (fSmooth) {
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_BLEND);
   }
   
   if (fStipple > 1)
      glDisable(GL_LINE_STIPPLE);
      
   if (fSetWidth)
      glLineWidth(1.f);
}

/*
Auxiliary class to draw markers in a gl-pad.
*/
//______________________________________________________________________________
void MarkerPainter::DrawDot(UInt_t n, const TPoint *xy)const
{
   //Simple 1-pixel dots.
   glBegin(GL_POINTS);
   
   for (UInt_t i = 0; i < n; ++i)
      glVertex2d(xy[i].fX, xy[i].fY);

   glEnd();
}

//______________________________________________________________________________
void MarkerPainter::DrawPlus(UInt_t n, const TPoint *xy)const
{
   //+ sign. 1 pixel width lines.
   const Double_t im = 4 * gVirtualX->GetMarkerSize() + 0.5;
   glBegin(GL_LINES);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      glVertex2d(-im + x, y);
      glVertex2d(im + x, y);
      glVertex2d(x, -im + y);
      glVertex2d(x, im + y);
   }
   
   glEnd();
}

//______________________________________________________________________________
void MarkerPainter::DrawStar(UInt_t n, const TPoint *xy)const
{
   //* - marker.
   SCoord_t im = SCoord_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   fStar[0].fX = -im;  fStar[0].fY = 0;
   fStar[1].fX =  im;  fStar[1].fY = 0;
   fStar[2].fX = 0  ;  fStar[2].fY = -im;
   fStar[3].fX = 0  ;  fStar[3].fY = im;
   im = SCoord_t(0.707*Float_t(im) + 0.5);
   fStar[4].fX = -im;  fStar[4].fY = -im;
   fStar[5].fX =  im;  fStar[5].fY = im;
   fStar[6].fX = -im;  fStar[6].fY = im;
   fStar[7].fX =  im;  fStar[7].fY = -im;
   
   glBegin(GL_LINES);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glVertex2d(fStar[0].fX + x, fStar[0].fY + y);
      glVertex2d(fStar[1].fX + x, fStar[1].fY + y);
      glVertex2d(fStar[2].fX + x, fStar[2].fY + y);
      glVertex2d(fStar[3].fX + x, fStar[3].fY + y);
      glVertex2d(fStar[4].fX + x, fStar[4].fY + y);
      glVertex2d(fStar[5].fX + x, fStar[5].fY + y);
      glVertex2d(fStar[6].fX + x, fStar[6].fY + y);
      glVertex2d(fStar[7].fX + x, fStar[7].fY + y);
   }
   
   glEnd();
}

//______________________________________________________________________________
void MarkerPainter::DrawX(UInt_t n, const TPoint *xy)const
{
   const Double_t im = 0.707 * (4 * gVirtualX->GetMarkerSize() + 0.5) + 0.5;

   glBegin(GL_LINES);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glVertex2d(-im + x, -im + y);
      glVertex2d(im + x, im + y);
      glVertex2d(-im + x, im + y);
      glVertex2d(im + x, -im + y);
   }
   
   glEnd();
}

//______________________________________________________________________________
void MarkerPainter::DrawFullDotSmall(UInt_t n, const TPoint *xy)const
{
   glBegin(GL_LINES);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glVertex2d(-1. + x, y);
      glVertex2d(x + 1., y);
      glVertex2d(x, -1. + y);
      glVertex2d(x, 1. + y);
   }
   
   glEnd();
}

//______________________________________________________________________________
void MarkerPainter::DrawFullDotMedium(UInt_t n, const TPoint *xy)const
{
   for (UInt_t i = 0; i < n; ++i)
      glRectd(xy[i].fX - 1, xy[i].fY - 1, xy[i].fX + 1, xy[i].fY + 1);
}

namespace {
//Auxilary function for MarkerPainter. Define near the end of this source file.
void CalculateCircle(std::vector<TPoint> &circle, Double_t r, UInt_t pts);
}

//______________________________________________________________________________
void MarkerPainter::DrawCircle(UInt_t n, const TPoint *xy)const
{
   Double_t r = 4 * gVirtualX->GetMarkerSize() + 0.5;
   if (r > 100.)
      r = 100.;//as in TGX11.
   
   fCircle.clear();
   CalculateCircle(fCircle, r, r < 100. ? kSmallCirclePts : kLargeCirclePts);
      
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glBegin(GL_LINE_LOOP);
      for (UInt_t j = 0, e = fCircle.size(); j < e; ++j)
         glVertex2d(fCircle[j].fX + x, fCircle[j].fY + y);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawFullDotLarge(UInt_t n, const TPoint *xy)const
{
   fCircle.clear();
   fCircle.push_back(TPoint(0, 0));
   
   Double_t r = 4 * gVirtualX->GetMarkerSize() + 0.5;
   if (r > 100.)
      r = 100;//as in TGX11.
      
   CalculateCircle(fCircle, r, r < 100 ? kSmallCirclePts : kLargeCirclePts);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glBegin(GL_TRIANGLE_FAN);
      for (UInt_t j = 0, e = fCircle.size(); j < e; ++j)
         glVertex2d(fCircle[j].fX + x, fCircle[j].fY + y);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawFullSquare(UInt_t n, const TPoint *xy)const
{
   const Double_t im = 4 * gVirtualX->GetMarkerSize() + 0.5;
   for (UInt_t i = 0; i < n; ++i)
      glRectd(xy[i].fX - im, xy[i].fY - im, xy[i].fX + im, xy[i].fY + im);
}

//______________________________________________________________________________
void MarkerPainter::DrawFullTrianlgeUp(UInt_t n, const TPoint *xy)const
{
   const Double_t im = 4 * gVirtualX->GetMarkerSize() + 0.5;
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      glBegin(GL_POLYGON);
      glVertex2d(x - im, y - im);
      glVertex2d(x + im, y - im);
      glVertex2d(x, im + y);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawFullTrianlgeDown(UInt_t n, const TPoint *xy)const
{
   const Int_t im = Int_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      glBegin(GL_POLYGON);
      glVertex2d(x - im, y + im);
      glVertex2d(x, y - im);
      glVertex2d(im + x, y + im);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawDiamond(UInt_t n, const TPoint *xy)const
{
   const Int_t im  = Int_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t imx = Int_t(2.66 * gVirtualX->GetMarkerSize() + 0.5);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
      
      glBegin(GL_LINE_LOOP);
      glVertex2d(x - imx,  y);
      glVertex2d(x, y - im);
      glVertex2d(x + imx, y);
      glVertex2d(x, y + im);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawCross(UInt_t n, const TPoint *xy)const
{
   const Int_t im  = Int_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t imx = Int_t(1.33 * gVirtualX->GetMarkerSize() + 0.5);

   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;
   
      glBegin(GL_LINE_LOOP);
      glVertex2d(x - im, y - imx);
      glVertex2d(x - imx, y - imx);
      glVertex2d(x - imx, y - im);
      glVertex2d(x + imx, y - im);
      glVertex2d(x + imx, y - imx);
      glVertex2d(x + im, y - imx);
      glVertex2d(x + im, y + imx);
      glVertex2d(x + imx, y + imx);
      glVertex2d(x + imx, y + im);
      glVertex2d(x - imx, y + im);
      glVertex2d(x - imx, y + imx);
      glVertex2d(x - im, y + imx);
      glEnd();
   }
}

//______________________________________________________________________________
void MarkerPainter::DrawFullStar(UInt_t n, const TPoint *xy)const
{
   // HIGZ full star pentagone
   const Int_t im  = Int_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im1 = Int_t(0.66 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im2 = Int_t(2.00 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im3 = Int_t(2.66 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im4 = Int_t(1.33 * gVirtualX->GetMarkerSize() + 0.5);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;

      glBegin(GL_TRIANGLES);
      glVertex2d(x - im, y - im4);//0
      glVertex2d(x - im2, y + im1);//1
      glVertex2d(x - im4, y - im4);//9
      
      glVertex2d(x - im2, y + im1);//1
      glVertex2d(x - im3, y + im);//2
      glVertex2d(x, y + im2);//3
      
      glVertex2d(x, y + im2);//3
      glVertex2d(x + im3, y + im);//4
      glVertex2d(x + im2, y + im1);//5
      
      glVertex2d(x + im2, y + im1);//5
      glVertex2d(x + im, y - im4);//6
      glVertex2d(x + im4, y - im4);//7
      
      glVertex2d(x + im4, y - im4);//7
      glVertex2d(x, y - im);//8
      glVertex2d(x - im4, y - im4);//9
      
      glVertex2d(x - im4, y - im4);//9
      glVertex2d(x - im2, y + im1);//1
      glVertex2d(x, y + im2);//3
      
      glVertex2d(x - im4, y - im4);//9
      glVertex2d(x, y + im2);//3
      glVertex2d(x + im2, y + im1);//5
      
      glVertex2d(x - im4, y - im4);//9
      glVertex2d(x + im2, y + im1);//5
      glVertex2d(x + im4, y - im4);//7
      
      glEnd();

   }
}

//______________________________________________________________________________
void MarkerPainter::DrawOpenStar(UInt_t n, const TPoint *xy)const
{
   // HIGZ full star pentagone
   const Int_t im  = Int_t(4 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im1 = Int_t(0.66 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im2 = Int_t(2.00 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im3 = Int_t(2.66 * gVirtualX->GetMarkerSize() + 0.5);
   const Int_t im4 = Int_t(1.33 * gVirtualX->GetMarkerSize() + 0.5);
   
   for (UInt_t i = 0; i < n; ++i) {
      const Double_t x = xy[i].fX;
      const Double_t y = xy[i].fY;

      glBegin(GL_LINE_LOOP);
      glVertex2d(x - im, y - im4);
      glVertex2d(x - im2, y + im1);
      glVertex2d(x - im3, y + im);
      glVertex2d(x, y + im2);
      glVertex2d(x + im3, y + im);
      glVertex2d(x + im2, y + im1);
      glVertex2d(x + im, y - im4);
      glVertex2d(x + im4, y - im4);
      glVertex2d(x, y - im);
      glVertex2d(x - im4, y - im4);
      glEnd();
   }
}

/*
Small RAII class for GLU tesselator.
*/
#ifndef CALLBACK
#define CALLBACK
#endif

extern "C" {
#if defined(__APPLE_CC__) && __APPLE_CC__ > 4000 && __APPLE_CC__ < 5450 && !defined(__INTEL_COMPILER)
    typedef GLvoid (*tess_t)(...);
#elif defined( __mips ) || defined( __linux__ ) || defined( __FreeBSD__ ) || defined( __OpenBSD__ ) || defined( __sun ) || defined (__CYGWIN__) || defined (__APPLE__)
    typedef GLvoid (*tess_t)();
#elif defined ( WIN32)
    typedef GLvoid (CALLBACK *tess_t)( );
#else
    #error "Error - need to define type tess_t for this platform/compiler"
#endif
}

//______________________________________________________________________________
void Begin(Int_t type)
{
   Tesselation_t *dump = Tesselator::GetDump();
   if (!dump)
      return;

   dump->push_back(MeshPatch_t(type));
}

//______________________________________________________________________________
void Vertex(const Double_t *v)
{
   Tesselation_t *dump = Tesselator::GetDump();
   if (!dump)
      return;

   std::vector<Double_t> & vs = dump->back().fPatch;
   vs.push_back(v[0]);
   vs.push_back(v[1]);
   vs.push_back(v[2]);
}

//______________________________________________________________________________
void End()
{
}

Tesselation_t *Tesselator::fVs = 0;

//______________________________________________________________________________
Tesselator::Tesselator(Bool_t dump)
               : fTess(0)
{
   GLUtesselator *tess = gluNewTess();
   if (!tess)
      throw std::runtime_error("tesselator creation failed");

   if (!dump) {
      gluTessCallback(tess, (GLenum)GLU_BEGIN,  (tess_t) glBegin);
      gluTessCallback(tess, (GLenum)GLU_END,    (tess_t) glEnd);
      gluTessCallback(tess, (GLenum)GLU_VERTEX, (tess_t) glVertex3dv);
   } else {
      gluTessCallback(tess, (GLenum)GLU_BEGIN,  (tess_t) Begin);
      gluTessCallback(tess, (GLenum)GLU_END,    (tess_t) End);
      gluTessCallback(tess, (GLenum)GLU_VERTEX, (tess_t) Vertex);
   }

   gluTessProperty(tess, GLU_TESS_TOLERANCE, 1e-10);
   fTess = tess;
}

//______________________________________________________________________________
Tesselator::~Tesselator()
{
   gluDeleteTess((GLUtesselator *)fTess);
}

/*
In future, this should be an interface to per-pad FBO.
Currently, in only save sizes and coordinates (?)
*/
//______________________________________________________________________________
OffScreenDevice::OffScreenDevice(UInt_t w, UInt_t h, UInt_t x, UInt_t y, Bool_t top)
                   : fW(w), fH(h), fX(x), fY(y), fTop(top)
{
}

//______________________________________________________________________________
GLLimits::GLLimits()
            : fMaxLineWidth(0.),
              fMaxPointSize(0.)
{
}

//______________________________________________________________________________
Double_t GLLimits::GetMaxLineWidth()const
{
   if (!fMaxLineWidth) {
      Double_t lp[2] = {};
      glGetDoublev(lineWidthPNAME, lp);//lineWidthPNAME is defined at the top of this file.
      fMaxLineWidth = lp[1];
   }
   
   return fMaxLineWidth;
}

//______________________________________________________________________________
Double_t GLLimits::GetMaxPointSize()const
{
   if (!fMaxPointSize) {
      Double_t lp[2] = {};
      glGetDoublev(pointSizePNAME, lp);//pointSizePNAME is defined at the top of this file.
      fMaxPointSize = lp[1];
   }
   
   return fMaxLineWidth;
}


//______________________________________________________________________________
void ExtractRGB(Color_t colorIndex, Float_t *rgb)
{
   const TColor *color = gROOT->GetColor(colorIndex);
   if (color)
      color->GetRGB(rgb[0], rgb[1], rgb[2]);
}

namespace {

//______________________________________________________________________________
void CalculateCircle(std::vector<TPoint> &circle, Double_t r, UInt_t pts)
{
   const Double_t delta = TMath::TwoPi() / pts;
   const UInt_t first = circle.size();
   Double_t angle = 0.;
   circle.resize(circle.size() + pts + 1);
   
   for (UInt_t i = 0; i < pts; ++i, angle += delta) {
      circle[first + i].fX = SCoord_t(r * TMath::Cos(angle));
      circle[first + i].fY = SCoord_t(r * TMath::Sin(angle));
   }
   
   circle.back().fX = circle[first].fX;
   circle.back().fY = circle[first].fY;
}

}//anonymous namespace

}//namespace Pad
}//namespace Rgl
