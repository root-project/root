// @(#)root/gl:$Name:  $:$Id: TF2GL.h,v 1.1 2007/06/23 21:23:21 brun Exp $
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

#ifndef ROOT_TGLObject
#include "TGLObject.h"
#endif
#ifndef ROOT_TGLPlotPainter
#include "TGLPlotPainter.h"
#endif

class TGLRnrCtx;
class TGLParametricEquation;
class TH2;


class TGLParametricEquationGL : public TGLObject
{
private:
   TGLParametricEquationGL(const TGLParametricEquationGL&);            // Not implemented
   TGLParametricEquationGL& operator=(const TGLParametricEquationGL&); // Not implemented

protected:
   TGLParametricEquation  *fM;
   TGLPlotPainter         *fPlotPainter;

   virtual void DirectDraw(TGLRnrCtx & rnrCtx) const;

public:
   TGLParametricEquationGL();
   virtual ~TGLParametricEquationGL();

   virtual Bool_t SetModel(TObject* obj, const Option_t* opt=0);
   virtual void   SetBBox();

   virtual Bool_t KeepDuringSmartRefresh() const { return kFALSE; }

   // To support two-level selection
   // virtual Bool_t SupportsSecondarySelect() const { return kTRUE; }
   // virtual void ProcessSelection(UInt_t* ptr, TGLViewer*, TGLScene*);

   ClassDef(TGLParametricEquationGL, 0) // GL renderer for TGLParametricEquation
};

#endif
