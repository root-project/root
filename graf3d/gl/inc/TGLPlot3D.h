// @(#)root/gl:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLPlot3D
#define ROOT_TGLPlot3D

#include "TGLObject.h"
#include "TGLPlotPainter.h"

class TVirtualPad;
class TPolyMarker3D;
class TH3;

class TGLPlot3D : public TGLObject
{
private:
   TGLPlot3D(const TGLPlot3D&);            // Not implemented
   TGLPlot3D& operator=(const TGLPlot3D&); // Not implemented

protected:
   TGLPlotPainter     *fPlotPainter;
   TGLPlotCoordinates  fCoord;

   void SetPainter(TGLPlotPainter* p);

   static TGLPlot3D* InstantiatePlot(TObject* obj);

public:
   TGLPlot3D();
   virtual ~TGLPlot3D();

   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }

   static TGLPlot3D* CreatePlot(TH3 *h, TPolyMarker3D *pm);
   static TGLPlot3D* CreatePlot(TObject* obj, const Option_t* opt, TVirtualPad* pad);
   static TGLPlot3D* CreatePlot(TObject* obj, const Option_t* opt, Bool_t logx, Bool_t logy, Bool_t logz);

   ClassDef(TGLPlot3D, 0); // Short description.
};

#endif
