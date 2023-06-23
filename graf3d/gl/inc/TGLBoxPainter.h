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

#include "TGLPlotPainter.h"
#include "TGLQuadric.h"
#include "TString.h"
#include "TGLUtil.h"

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

   char   *GetPlotInfo(Int_t px, Int_t py) override;
   Bool_t  InitGeometry() override;
   void    StartPan(Int_t px, Int_t py) override;
   void    Pan(Int_t px, Int_t py) override;
   void    AddOption(const TString &stringOption) override;
   void    ProcessEvent(Int_t event, Int_t px, Int_t py) override;

private:
   //Overriders
   void    InitGL()const override;
   void    DeInitGL()const override;

   void    DrawPlot()const override;
   //Special type of TH3:
   void    DrawCloud()const;

   void    SetPlotColor()const;

   void    DrawSectionXOZ()const override;
   void    DrawSectionYOZ()const override;
   void    DrawSectionXOY()const override;

   void    DrawPalette()const;
   void    DrawPaletteAxis()const override;

   Bool_t  HasSections()const;

   ClassDefOverride(TGLBoxPainter, 0)//Box painter
};

#endif
