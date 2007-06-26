// @(#)root/gl:$Name:  $:$Id: TF2GL.cxx,v 1.1 2007/06/23 21:23:21 brun Exp $
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLParametricEquationGL.h"

#include "TGLParametric.h"
#include "TVirtualPad.h"

#include "TGLSurfacePainter.h"
#include "TGLTF3Painter.h"
#include "TGLAxis.h"

#include "TGLRnrCtx.h"

#include "TGLIncludes.h"

//______________________________________________________________________
// TGLParametricEquationGL
//
//

ClassImp(TGLParametricEquationGL)

//______________________________________________________________________________
TGLParametricEquationGL::TGLParametricEquationGL() : TGLObject(), fM(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

//______________________________________________________________________________
TGLParametricEquationGL::~TGLParametricEquationGL()
{
   // Destructor.

   delete fPlotPainter;
}

//______________________________________________________________________________
Bool_t TGLParametricEquationGL::SetModel(TObject* obj, const Option_t* opt)
{
   // Set model object.

   if(SetModelCheckClass(obj, TGLParametricEquation::Class()))
   {
      fM = dynamic_cast<TGLParametricEquation*>(obj);
      fPlotPainter = new TGLParametricPlot(fM, 0);
      TString option(opt);
      fPlotPainter->AddOption(option);
      fPlotPainter->InitGeometry();
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGLParametricEquationGL::SetBBox()
{
   // Setup bounding-box.

   fBoundingBox.Set(fPlotPainter->RefBackBox().Get3DBox());
}

//______________________________________________________________________________
void TGLParametricEquationGL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
{
   // Render the object.

   fPlotPainter->RefBackBox().FindFrontPoint();
   glPushAttrib(GL_ENABLE_BIT | GL_LIGHTING_BIT);
   glEnable(GL_NORMALIZE);
   fPlotPainter->InitGL();
   fPlotPainter->DrawPlot();
   glPopAttrib();
}
