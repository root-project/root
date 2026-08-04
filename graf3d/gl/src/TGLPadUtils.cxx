// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  06/05/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <stdexcept>
#include <cassert>
#include <cmath>

#include "TVirtualX.h"
#include "RStipples.h"
#include "TColor.h"
#include "TROOT.h"
#include "TMath.h"
#include "TAttMarker.h"

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

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

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
////////////////////////////////////////////////////////////////////////////////
///Polygon stipple, if required.

FillAttribSet::FillAttribSet(const PolygonStippleSet &set, Bool_t ignoreStipple, const TAttFill *att)
                  : fStipple(0), fAlpha(1.)
{
   Style_t fillStyle = att ? att->GetFillStyle() : gVirtualX->GetFillStyle();
   Color_t fillColor = att ? att->GetFillColor() : gVirtualX->GetFillColor();

   const UInt_t style = fillStyle / 1000;

   if (!ignoreStipple) {
      if (style == 3) {
         const UInt_t fasi  = fillStyle % 1000;
         fStipple = (fasi >= 1 && fasi <=25) ? fasi : 2;
         glPolygonStipple(&set.fStipples[fStipple * PolygonStippleSet::kStippleSize]);
         glEnable(GL_POLYGON_STIPPLE);
      }
   }

   // Color and transparency
   Float_t rgba[] = {0.f, 0.f, 0.f, 1.f};
   ExtractRGBA(fillColor, rgba);
   fAlpha = rgba[3];
   if (fAlpha<1.) {
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }
   glColor4fv(rgba);
}

////////////////////////////////////////////////////////////////////////////////

FillAttribSet::~FillAttribSet()
{
   if (fStipple)
      glDisable(GL_POLYGON_STIPPLE);

   if (fAlpha<1.)
      glDisable(GL_BLEND);
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
////////////////////////////////////////////////////////////////////////////////
///Set up line parameters.
///Smooth.

LineAttribSet::LineAttribSet(Bool_t smooth, UInt_t stipple, Double_t maxWidth, Bool_t setWidth, const TAttLine *att)
                  : fSmooth(smooth), fStipple(stipple), fSetWidth(setWidth), fAlpha(0.8)
{
   if (fSmooth) {
      glEnable(GL_BLEND);
      glEnable(GL_LINE_SMOOTH);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
   }

   Color_t lineColor = att ? att->GetLineColor() : gVirtualX->GetLineColor();
   Width_t lineWidth = att ? att->GetLineWidth() : gVirtualX->GetLineWidth();

   //Stipple.
   if (fStipple > 1) {
      if (fStipple >= gMaxStipple)
         fStipple = 1;
      else {
         glEnable(GL_LINE_STIPPLE);
         glLineStipple(fStipple == 10 ? 2 : 1, gLineStipples[fStipple]);
      }
   }

   //Color and transparency
   Float_t rgba[] = {0.f, 0.f, 0.f, 0.8f};
   ExtractRGBA(lineColor, rgba);
   fAlpha = rgba[3];
   if (fAlpha<0.8) {
      glEnable(GL_BLEND);
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
   }
   glColor4fv(rgba);

   //Width.
   if (fSetWidth) {
      glLineWidth(lineWidth > maxWidth ? maxWidth : !lineWidth ? 1.f : lineWidth);
   }
}

////////////////////////////////////////////////////////////////////////////////

LineAttribSet::~LineAttribSet()
{
   if (fSmooth || fAlpha<0.8) {
      glDisable(GL_LINE_SMOOTH);
      glDisable(GL_BLEND);
   }

   if (fStipple > 1)
      glDisable(GL_LINE_STIPPLE);

   if (fSetWidth)
      glLineWidth(1.f);
}


////////////////////////////////////////////////////////////////////////////////
/// Auxiliary class to draw markers in a gl-pad.

void MarkerPainter::DrawMarkers(UInt_t n, const TPoint *xy, const TAttMarker &attr)
{
   Int_t markerSize = 0;
   std::vector<TPoint> markerShape;
   auto markerType = attr.GetMarkerShape(markerSize, markerShape, 1., kTRUE);

   auto masrkerStyle = TAttMarker::GetMarkerStyleBase(attr.GetMarkerStyle());
   Bool_t changePolygonMode = (masrkerStyle == kOpenSquare) || (masrkerStyle == kOpenTriangleUp);

   switch(markerType) {
      case TAttMarker::kShapeDot:
         glBegin(GL_POINTS);
         for (UInt_t i = 0; i < n; ++i)
            glVertex2d(xy[i].fX, xy[i].fY);
         glEnd();
         break;
      case TAttMarker::kShapeFilledCircle:
         // to fill circle, place point in the middle
         markerShape.emplace_back(0,0);
         // no break, circle points will be append
      case TAttMarker::kShapeCircle: {
         Double_t r = markerSize * 0.5;
         const int pts = r < 100 ? kSmallCirclePts : kLargeCirclePts;
         const Double_t delta = TMath::TwoPi() / pts;
         markerShape.reserve(markerShape.size() + pts + 1);
         Double_t angle = 0.;
         for (int i = 0; i < pts; ++i, angle += delta)
            markerShape.emplace_back(std::round(r * TMath::Cos(angle)), std::round(r * TMath::Sin(angle)));
         markerShape.emplace_back(r, 0); // close circle
      }
      // no break, markerShape will be used as all other marker shapes
      case TAttMarker::kShapePolyLine:
      case TAttMarker::kShapeFilledArea:
         if (changePolygonMode)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
         for (unsigned i = 0; i < n; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            if ((markerType == TAttMarker::kShapePolyLine) || (markerType == TAttMarker::kShapeCircle))
               glBegin(GL_LINE_LOOP);
            else if (markerType == TAttMarker::kShapeFilledCircle)
               glBegin(GL_TRIANGLE_FAN);
            else
               glBegin(GL_POLYGON);

            for (auto &pnt : markerShape)
               glVertex2d(x + pnt.fX, y - pnt.fY);
            glEnd();
         }
         if (changePolygonMode)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
         break;
      case TAttMarker::kShapeSegments:
         glBegin(GL_LINES);
         for (unsigned i = 0; i < n; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            for (auto &pnt : markerShape)
               glVertex2d(x + pnt.fX, y - pnt.fY);
         }
         glEnd();
         break;
      case TAttMarker::kShapeTriangles:
         for (unsigned i = 0; i < n; ++i) {
            const Double_t x = xy[i].fX;
            const Double_t y = xy[i].fY;
            glBegin(GL_TRIANGLES);
            for (auto &pnt : markerShape)
               glVertex2d(x + pnt.fX, y - pnt.fY);
            glEnd();
         }
         break;
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

////////////////////////////////////////////////////////////////////////////////

void Begin(Int_t type)
{
   Tesselation_t *dump = Tesselator::GetDump();
   if (!dump)
      return;

   dump->push_back(MeshPatch_t(type));
}

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

void End()
{
}

Tesselation_t *Tesselator::fVs = nullptr;

////////////////////////////////////////////////////////////////////////////////

Tesselator::Tesselator(Bool_t dump)
               : fTess(nullptr)
{
   GLUtesselator *tess = gluNewTess();
   if (!tess)
      throw std::runtime_error("tesselator creation failed");

#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-function-type"
#endif

   if (!dump) {
      gluTessCallback(tess, (GLenum)GLU_BEGIN,  (tess_t) impl_glBegin);
      gluTessCallback(tess, (GLenum)GLU_END,    (tess_t) impl_glEnd);
      gluTessCallback(tess, (GLenum)GLU_VERTEX, (tess_t) impl_glVertex3dv);
   } else {
      gluTessCallback(tess, (GLenum)GLU_BEGIN,  (tess_t) Begin);
      gluTessCallback(tess, (GLenum)GLU_END,    (tess_t) End);
      gluTessCallback(tess, (GLenum)GLU_VERTEX, (tess_t) Vertex);
   }

#if defined(__GNUC__) && __GNUC__ >= 8
#pragma GCC diagnostic pop
#endif

   gluTessProperty(tess, GLU_TESS_TOLERANCE, 1e-10);
   fTess = tess;
}

////////////////////////////////////////////////////////////////////////////////

Tesselator::~Tesselator()
{
   gluDeleteTess((GLUtesselator *)fTess);
}

/*
In future, this should be an interface to per-pad FBO.
Currently, in only save sizes and coordinates (?)
*/
////////////////////////////////////////////////////////////////////////////////

OffScreenDevice::OffScreenDevice(UInt_t w, UInt_t h, UInt_t x, UInt_t y, Bool_t top)
                   : fW(w), fH(h), fX(x), fY(y), fTop(top)
{
}

////////////////////////////////////////////////////////////////////////////////

GLLimits::GLLimits()
            : fMaxLineWidth(0.),
              fMaxPointSize(0.)
{
}

////////////////////////////////////////////////////////////////////////////////

Double_t GLLimits::GetMaxLineWidth()const
{
   if (!fMaxLineWidth) {
      Double_t lp[2] = {};
      glGetDoublev(lineWidthPNAME, lp);//lineWidthPNAME is defined at the top of this file.
      fMaxLineWidth = lp[1];
   }

   return fMaxLineWidth;
}

////////////////////////////////////////////////////////////////////////////////

Double_t GLLimits::GetMaxPointSize()const
{
   if (!fMaxPointSize) {
      Double_t lp[2] = {};
      glGetDoublev(pointSizePNAME, lp);//pointSizePNAME is defined at the top of this file.
      fMaxPointSize = lp[1];
   }

   return fMaxLineWidth;
}


////////////////////////////////////////////////////////////////////////////////

void ExtractRGBA(Color_t colorIndex, Float_t *rgba)
{
   const TColor *color = gROOT->GetColor(colorIndex);
   if (color) {
      color->GetRGB(rgba[0], rgba[1], rgba[2]);
      rgba[3] = color->GetAlpha();
   }
}

////////////////////////////////////////////////////////////////////////////////

template<class ValueType>
BoundingRect<ValueType> FindBoundingRect(Int_t nPoints, const ValueType *xs, const ValueType *ys)
{
   assert(nPoints > 0 && "FindBoundingRect, invalind number of points");
   assert(xs != nullptr && "FindBoundingRect, parameter 'xs' is null");
   assert(ys != nullptr && "FindBoundingRect, parameter 'ys' is null");

   ValueType xMin = xs[0], xMax = xMin;
   ValueType yMin = ys[0], yMax = yMin;

   for (Int_t i = 1; i < nPoints; ++i) {
      xMin = TMath::Min(xMin, xs[i]);
      xMax = TMath::Max(xMax, xs[i]);

      yMin = TMath::Min(yMin, ys[i]);
      yMax = TMath::Max(yMax, ys[i]);
   }

   BoundingRect<ValueType> box = {};
   box.fXMin = xMin;
   box.fXMax = xMax;
   box.fWidth = xMax - xMin;

   box.fYMin = yMin;
   box.fYMax = yMax;
   box.fHeight = yMax - yMin;

   return box;
}

template BoundingRect<Double_t> FindBoundingRect(Int_t nPoints, const Double_t *xs, const Double_t *ys);
template BoundingRect<Float_t> FindBoundingRect(Int_t nPoints, const Float_t *xs, const Float_t *ys);
template BoundingRect<Long_t> FindBoundingRect(Int_t nPoints, const Long_t *xs, const Long_t *ys);
template BoundingRect<Int_t> FindBoundingRect(Int_t nPoints, const Int_t *xs, const Int_t *ys);
template BoundingRect<SCoord_t> FindBoundingRect(Int_t nPoints, const SCoord_t *xs, const SCoord_t *ys);


}//namespace Pad
}//namespace Rgl
