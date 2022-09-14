// @(#)root/eve:$Id$
// Author: Matevz Tadel, 2009

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEvePlot3DGL
#define ROOT_TEvePlot3DGL

#include "TGLObject.h"

class TGLViewer;
class TGLScene;

class TEvePlot3D;
class TGLPlot3D;

class TEvePlot3DGL : public TGLObject
{
private:
   TEvePlot3DGL(const TEvePlot3DGL&);            // Not implemented
   TEvePlot3DGL& operator=(const TEvePlot3DGL&); // Not implemented

protected:
   TEvePlot3D      *fM;           // Model object.
   TGLPlot3D       *fPlotLogical; // Actual painter.

public:
   TEvePlot3DGL();
   virtual ~TEvePlot3DGL() {}

   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }

   virtual Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr);
   virtual void   SetBBox();

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(TGLRnrCtx & rnrCtx, TGLSelectRecord & rec);

   ClassDef(TEvePlot3DGL, 0); // GL renderer class for TEvePlot3D.
};

#endif
