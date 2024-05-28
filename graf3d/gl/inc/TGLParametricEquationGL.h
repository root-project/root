// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGLParametricEquationGL
#define ROOT_TGLParametricEquationGL

#include "TGLPlot3D.h"

class TGLRnrCtx;
class TGLParametricEquation;
class TH2;


class TGLParametricEquationGL : public TGLPlot3D
{
private:
   TGLParametricEquationGL(const TGLParametricEquationGL&) = delete;
   TGLParametricEquationGL& operator=(const TGLParametricEquationGL&) = delete;

protected:
   TGLParametricEquation  *fM;

public:
   TGLParametricEquationGL();
   ~TGLParametricEquationGL() override;

   Bool_t SetModel(TObject* obj, const Option_t *opt = nullptr) override;
   void   SetBBox() override;
   void   DirectDraw(TGLRnrCtx & rnrCtx) const override;

   Bool_t KeepDuringSmartRefresh() const override { return kFALSE; }

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDefOverride(TGLParametricEquationGL, 0); // GL renderer for TGLParametricEquation
};

#endif
