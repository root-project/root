// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  26/01/2007

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLParametric
#define ROOT_TGLParametric

#include <memory>

#ifndef ROOT_TGLHistPainter
#include "TGLHistPainter.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif
#ifndef ROOT_TAxis
#include "TAxis.h"
#endif
#ifndef ROOT_TF2
#include "TF2.h"
#endif

class TString;

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGLParametricEquation                                                //
//                                                                      //
// Parametric equations drawing with GL.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


typedef void (*ParametricEquation_t)(TGLVertex3 &, Double_t u, Double_t v);

class TGLParametricEquation : public TNamed {
private:
   typedef std::auto_ptr<TF2> Ptr_t;

   Ptr_t                fXEquation;
   Ptr_t                fYEquation;
   Ptr_t                fZEquation;

   ParametricEquation_t fEquation;

   Rgl::Range_t         fURange;
   Rgl::Range_t         fVRange;

   Bool_t               fConstrained;
   Bool_t               fModified;

   typedef std::auto_ptr<TGLHistPainter> Painter_t;
   //C++ compiler do not need TGLhistPainter definition here, but I'm not sure about CINT,
   //so I've included TGLHistPainter definition.
   Painter_t            fPainter;

public:
   TGLParametricEquation(const TString &name, const TString &xEquation,
                 const TString &yEquation, const TString &zEquation,
                 Double_t uMin, Double_t uMax,
                 Double_t vMin, Double_t vMax);
   TGLParametricEquation(const TString &name, ParametricEquation_t equation,
                 Double_t uMin, Double_t uMax, Double_t vMin, Double_t vMax);

   Rgl::Range_t GetURange()const;
   Rgl::Range_t GetVRange()const;

   Bool_t       IsConstrained()const;
   void         SetConstrained(Bool_t c);

   Bool_t       IsModified()const;
   void         SetModified(Bool_t m);

   void         EvalVertex(TGLVertex3 &newVertex, Double_t u, Double_t v)const;

   Int_t        DistancetoPrimitive(Int_t px, Int_t py);
   void         ExecuteEvent(Int_t event, Int_t px, Int_t py);
   char        *GetObjectInfo(Int_t px, Int_t py) const;
   void         Paint(Option_t *option);

private:

   TGLParametricEquation(const TGLParametricEquation &);
   TGLParametricEquation &operator = (const TGLParametricEquation &);

   ClassDef(TGLParametricEquation, 0)//Equation of parametric surface.
};

class TGLParametricPlot : public TGLPlotPainter {
private:
   struct Vertex_t {
      TGLVertex3 fPos;
      TGLVector3 fNormal;
      Float_t    fRGBA[4];
   };

   enum EMeshSize {kLow = 30, kHigh = 150};

   Int_t                  fMeshSize;
   TGL2DArray<Vertex_t>   fMesh;

   Bool_t                 fShowMesh;
   Int_t                  fColorScheme;

   TGLParametricEquation *fEquation;

   TAxis                  fCartesianXAxis;
   TAxis                  fCartesianYAxis;
   TAxis                  fCartesianZAxis;

   TGLPlotCoordinates     fCartesianCoord;

public:
   TGLParametricPlot(TGLParametricEquation *equation, TGLPlotCamera *camera);

   Bool_t   InitGeometry();
   void     StartPan(Int_t px, Int_t py);
   void     Pan(Int_t px, Int_t py);
   char    *GetPlotInfo(Int_t px, Int_t py);
   void     AddOption(const TString &option);
   void     ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   void     InitGL()const;
   void     DeInitGL()const;

   void     DrawPlot()const;

   void     InitColors();

   void     DrawSectionXOZ()const;
   void     DrawSectionYOZ()const;
   void     DrawSectionXOY()const;

   void     SetSurfaceColor()const;

   TGLParametricPlot(const TGLParametricPlot &);
   TGLParametricPlot &operator = (const TGLParametricPlot &);

   ClassDef(TGLParametricPlot, 0)//Parametric plot's painter.
};

#endif
