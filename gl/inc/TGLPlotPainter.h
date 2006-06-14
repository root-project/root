// @(#)root/gl:$Name:  $:$Id: TGLPainterAlgorithms.h,v 1.1 2006/06/14 10:00:00 couet Exp $
// Author:  Timur Pocheptsov  14/06/2006
                                                                                
/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPainterAlgorithms
#define ROOT_TGLPainterAlgorithms

#include <utility>

#ifndef ROOT_TVirtualGL
#include "TVirtualGL.h"
#endif
#ifndef ROOT_TArcBall
#include "TArcBall.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TPoint
#include "TPoint.h"
#endif

class TString;
class TColor;
class TAxis;
class TH1;

/*
   TGLPlotPainter class defines interface to different plot painters.
*/

enum EGLCoordType {
   kGLCartesian,
   kGLPolar,
   kGLCylindrical,
   kGLSpherical
};

enum EGLPlotType {
   kGLLegoPlot,
   kGLSurfacePlot,
   kGLBoxPlot,
   kGLTF3Plot,
   kGLStackPlot,
   kGLDefaultPlot
};

class TGLPlotPainter : public TVirtualGLPainter {
public:
   virtual void     SetGLContext(Int_t context) = 0;
   //Shows info about an object under cursor.
   virtual char    *GetObjectInfo(Int_t px, Int_t py) = 0;
   //Init geometry does plot's specific initialization.
   //Such initialization can be done one time in constructor,
   //without any InitGeometry call. But in case of pad,
   //user can change something interactivly and
   //painter should know about it to change its state.
   virtual Bool_t   InitGeometry() = 0;
   virtual void     StartRotation(Int_t px, Int_t py) = 0;
   virtual void     Rotate(Int_t px, Int_t py) = 0;
   virtual void     StopRotation() = 0;
   virtual void     StartPan(Int_t px, Int_t py) = 0;
   //Pan function is already declared in TVirtualGLPainter
   virtual void     StopPan() = 0;
   virtual void     ZoomIn() = 0;
   virtual void     ZoomOut() = 0;
   virtual void     SetLogX(Bool_t logX) = 0;
   virtual void     SetLogY(Bool_t logY) = 0;
   virtual void     SetLogZ(Bool_t logZ) = 0;
   virtual void     SetCoordType(EGLCoordType type) = 0;
   //Add string option, it can be a digit in "lego" or "surf"
   //options, and it can be some others options - "e".
   virtual void     AddOption(const TString &stringOption) = 0;
   //Used by GLpad
   virtual void     SetPadColor(TColor *color) = 0;
   //Used by GLpad
   virtual void     SetFrameColor(TColor *color) = 0;
   //Function to process additional events (some key presses etc.)
   virtual void     ProcessEvent(Int_t event, Int_t px, Int_t py) = 0;

   ClassDef(TGLPlotPainter, 0) //Base for gl plots
};

/*
   Auxilary class, which holds info about plot's back box and
   sizes.
*/

class TGLPlotFrame {
   friend class TGL2DAxisPainter;

protected:
   typedef std::pair<Double_t, Double_t> Range_t;
   typedef std::pair<Int_t, Int_t>       BinRange_t;

   TGLVertex3            fFrame[8];
   static const Int_t    fFramePlanes[][4];
   static const Double_t fFrameNormals[][3];
   static const Int_t    fBackPairs[][2];

   Bool_t          fLogX;
   Bool_t          fLogY;
   Bool_t          fLogZ;
   Double_t        fScaleX;
   Double_t        fScaleY;
   Double_t        fScaleZ;

   Int_t           fFrontPoint;
   TGLVertex3      f2DAxes[8];

   TArcBall        fArcBall;
   Int_t           fViewport[4];

   TPoint          fMousePosition;
   TGLVector3      fPan;

   Double_t        fZoom;
   Double_t        fFrustum[4];
   Double_t        fShift;

   Double_t        fCenter[3];
   Double_t        fFactor;

   TGLPlotFrame(Bool_t logX, Bool_t logY, Bool_t logZ);
   virtual ~TGLPlotFrame();

   void CalculateGLCameraParams(const Range_t &x, const Range_t &y, const Range_t &z);
   void FindFrontPoint();
   void SetTransformation();
   void SetCamera();

protected:
   static Bool_t ExtractAxisInfo(const TAxis *axis, Bool_t log, BinRange_t &bins, Range_t &range);
   Bool_t ExtractAxisZInfo(TH1 *hist, Bool_t logZ, const BinRange_t &xBins, 
                           const BinRange_t &yBins, Range_t &zRange);
   static void   AdjustShift(const TPoint &start, const TPoint &finish, TGLVector3 &shiftVec, const Int_t *viewport);

   ClassDef(TGLPlotFrame, 0) //Auxilary class
};

class TGLQuadric;

namespace RootGL
{

   void DrawCylinder(TGLQuadric *quadric, Double_t xMin, Double_t xMax, Double_t yMin, 
                     Double_t yMax, Double_t zMin, Double_t zMax);
   void DrawQuadOutline(const TGLVertex3 &v1, const TGLVertex3 &v2, 
                        const TGLVertex3 &v3, const TGLVertex3 &v4);
   void DrawQuadFilled(const TGLVertex3 &v0, const TGLVertex3 &v1, const TGLVertex3 &v2,
                       const TGLVertex3 &v3, const TGLVertex3 &normal);
   void DrawBoxFront(Double_t xMin, Double_t xMax, Double_t yMin, Double_t yMax, 
                     Double_t zMin, Double_t zMax, Int_t frontPoint);
   void DrawBoxFrontTextured(Double_t x1, Double_t x2, Double_t y1, Double_t y2, Double_t z1, 
                             Double_t z2, Double_t texMin, Double_t texMax, Int_t frontPoint);
   void DrawTrapezoid(const Double_t ver[][2], Double_t zMin, Double_t zMax, Bool_t needNormals = kTRUE);
   void DrawTrapezoid(const Double_t ver[][3]);
   void DrawTrapezoidTextured(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                              Double_t texMin, Double_t texMax);
   void DrawTrapezoidTextured(const Double_t ver[][3], Double_t texMin, Double_t texMax);
   void DrawTrapezoidTextured2(const Double_t ver[][2], Double_t zMin, Double_t zMax,
                               Double_t texMin, Double_t texMax);
}

#endif
