// @(#)root/gl:$Name:  $:$Id: TF2GL.cxx,v 1.1 2007/06/23 21:23:21 brun Exp $
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TF2GL.h"

#include <TF2.h>
#include <TF3.h>
#include <TH2.h>
#include <TVirtualPad.h>

#include "TGLSurfacePainter.h"
#include "TGLTF3Painter.h"
#include "TGLAxis.h"

#include "TGLRnrCtx.h"

#include "TGLIncludes.h"

//______________________________________________________________________
// TF2GL
//
//

ClassImp(TF2GL)

TF2GL::TF2GL() : TGLObject(), fM(0), fH(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

TF2GL::~TF2GL()
{
   // Destructor.

   delete fH;
   delete fPlotPainter;
}

/**************************************************************************/

Bool_t TF2GL::SetModel(TObject* obj, const Option_t* opt)
{
   // Set model object.

   if(SetModelCheckClass(obj, TF2::Class()))
   {
      fM = dynamic_cast<TF2*>(obj);
      fH = (TH2*) fM->CreateHistogram();

      if (dynamic_cast<TF3*>(fM))
         fPlotPainter = new TGLTF3Painter((TF3*)fM, fH, 0, &fCoord);
      else
         fPlotPainter = new TGLSurfacePainter(fH, 0, &fCoord);

      // Coord-system
      fCoord.SetXLog(gPad->GetLogx());
      fCoord.SetYLog(gPad->GetLogy());
      fCoord.SetZLog(gPad->GetLogz());

      TString option(opt);

      if (option.Index("sph") != kNPOS)
         fCoord.SetCoordType(kGLSpherical);
      else if (option.Index("pol") != kNPOS)
         fCoord.SetCoordType(kGLPolar);
      else if (option.Index("cyl") != kNPOS)
         fCoord.SetCoordType(kGLCylindrical);
      
      fPlotPainter->AddOption(option);

      fPlotPainter->InitGeometry();
      return kTRUE;
   }
   return kFALSE;
}

void TF2GL::SetBBox()
{
   // Setup bounding-box.

   fBoundingBox.Set(fPlotPainter->RefBackBox().Get3DBox());
}

/**************************************************************************/

void TF2GL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
{
   // Render the object.

   fPlotPainter->RefBackBox().FindFrontPoint();
   glPushAttrib(GL_ENABLE_BIT | GL_LIGHTING_BIT);
   glEnable(GL_NORMALIZE);
   fPlotPainter->InitGL();
   fPlotPainter->DrawPlot();

   glDisable(GL_CULL_FACE);

   TGLAxis ap;
   const Rgl::Range_t & xr = fCoord.GetXRange();
   ap.PaintGLAxis(fBoundingBox[0].CArr(), fBoundingBox[1].CArr(),
                  xr.first, xr.second, 205);
   const Rgl::Range_t & yr = fCoord.GetXRange();
   ap.PaintGLAxis(fBoundingBox[0].CArr(), fBoundingBox[3].CArr(),
                  yr.first, yr.second, 205);
   const Rgl::Range_t & zr = fCoord.GetXRange();
   ap.PaintGLAxis(fBoundingBox[0].CArr(), fBoundingBox[4].CArr(),
                  zr.first, zr.second, 205);
                  
   glPopAttrib();
}
