// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  31/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TGLBoxPainter
#define ROOT_TGLBoxPainter

#include <vector>

#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif
#ifndef ROOT_TGLQuadric
#include "TGLQuadric.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TGLUtil
#include "TGLUtil.h"
#endif

//TGLScenePad creates box painter
//for the case TPad has poly marker and
//empty TH3 inside.
//Now it's up to box painter to do everything
//correctly.
class TPolyMarker3D;

class TGLPlotCamera;
class TAxis;
class TH1;

class TGLBoxPainter : public TGLPlotPainter {
private:
   TGLTH3Slice fXOZSlice;
   TGLTH3Slice fYOZSlice;
   TGLTH3Slice fXOYSlice;

   enum EBoxType {
      kBox, //boxes, sizes are proportional to bin content
      kBox1 //spheres, not boxes
   };

   mutable EBoxType        fType;

   TString                 fPlotInfo;
   Rgl::Range_t            fMinMaxVal;

   mutable TGLQuadric      fQuadric;

   const TPolyMarker3D    *fPolymarker; //Polymarker from TTree.
   std::vector<Double_t>   fPMPoints;   //Cache for polymarker's points.

   TGLBoxPainter(const TGLBoxPainter &);
   TGLBoxPainter &operator = (const TGLBoxPainter &);

public:
   TGLBoxPainter(TH1 *hist, TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   TGLBoxPainter(TH1 *hist, TPolyMarker3D * pm,
                 TGLPlotCamera *camera, TGLPlotCoordinates *coord);

   char   *GetPlotInfo(Int_t px, Int_t py);
   Bool_t  InitGeometry();
   void    StartPan(Int_t px, Int_t py);
   void    Pan(Int_t px, Int_t py);
   void    AddOption(const TString &stringOption);
   void    ProcessEvent(Int_t event, Int_t px, Int_t py);

private:
   //Overriders
   void    InitGL()const;
   void    DeInitGL()const;

   void    DrawPlot()const;
   //Special type of TH3:
   void    DrawCloud()const;

   void    SetPlotColor()const;

   void    DrawSectionXOZ()const;
   void    DrawSectionYOZ()const;
   void    DrawSectionXOY()const;

   void    DrawPalette()const;
   void    DrawPaletteAxis()const;

   Bool_t  HasSections()const;

   ClassDef(TGLBoxPainter, 0)//Box painter
};

#endif
