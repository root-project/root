// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  17/11/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLHistPainter
#define ROOT_TGLHistPainter

#include "TVirtualHistPainter.h"
#include "TGLPlotPainter.h"
#include "TGLPlotCamera.h"

#include <memory>

/*
   TGLHistPainter is a proxy class. It inherits TVirtualHistPainter and
   overrides its virtual functions, but all actual work is done by :
      THistPainter - I name it "default" painter, it's the member of type
                     TVirtualHistPainter * and loaded via plugin-manager;
      TGLLegoPainter - it draws different legos (lego/lego1/lego2/lego3);
      TGLSurfacePainter - supports surfaces (surf/surf1/surf2/surf3/surf4/surf5);
      TGLBoxPainter - box option for TH3;
      TGLTF3Painter - TF3.
*/

class TGLParametricEquation;
class TGLTH3Composition;
class TGL5DDataSet;
class TString;
class TList;
class TF3;
class TH1;

class TGLHistPainter : public TVirtualHistPainter {
private:
   //Dynamic type is THistPainter, no problems with simultaneous inheritance and membership
   //TGLHistPainter delegates unsupported options/calls to this object
   std::unique_ptr<TVirtualHistPainter> fDefaultPainter;
   //This member can have different dynamic types: TGLLegoPainter, etc.
   std::unique_ptr<TGLPlotPainter>      fGLPainter;

   TGLParametricEquation *fEq;
   TH1                   *fHist;
   TF3                   *fF3;
   TList                 *fStack;
   EGLPlotType            fPlotType;
   TGLPlotCamera          fCamera;
   TGLPlotCoordinates     fCoord;

public:
   TGLHistPainter(TH1 *hist);
   TGLHistPainter(TGLParametricEquation *equation);
   TGLHistPainter(TGL5DDataSet *data);
   TGLHistPainter(TGLTH3Composition *comp);

   //TVirtualHistPainter final overriders
   Int_t          DistancetoPrimitive(Int_t px, Int_t py);
   void           DrawPanel();
   void           ExecuteEvent(Int_t event, Int_t px, Int_t py);
   TList         *GetContourList(Double_t contour)const;
   char          *GetObjectInfo(Int_t px, Int_t py)const;
   TList         *GetStack()const;
   Bool_t         IsInside(Int_t x, Int_t y);
   Bool_t         IsInside(Double_t x, Double_t y);
   void           Paint(Option_t *option);
   void           PaintStat(Int_t dostat, TF1 *fit);
   void           ProcessMessage(const char *message, const TObject *obj);
   void           SetHighlight();
   void           SetHistogram(TH1 *hist);
   void           SetStack(TList *stack);
   Int_t          MakeCuts(char *cutsOpt);
   void           SetShowProjection(const char *option, Int_t nbins);

   TGLPlotPainter *GetRealPainter(){return fGLPainter.get();}
private:

   struct PlotOption_t;

   PlotOption_t   ParsePaintOption(const TString &option)const;
   void           CreatePainter(const PlotOption_t &parsed,
                                const TString &option);

   void           PadToViewport(Bool_t selectionPass = kFALSE);

   TGLHistPainter(const TGLHistPainter &);
   TGLHistPainter &operator = (const TGLHistPainter &);

   ClassDef(TGLHistPainter, 0) //Proxy class for GL hist painters.
};

#endif
