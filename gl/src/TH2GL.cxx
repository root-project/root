// @(#)root/gl:$Name$:$Id$
// Author:  Matevz Tadel, Jun 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TH2GL.h"
#include <TH2.h>
#include "TVirtualPad.h"

#include "TGLSurfacePainter.h"
#include "TGLHistPainter.h"
#include "TGLLegoPainter.h"
#include "TGLBoxPainter.h"
#include "TGLTF3Painter.h"
#include "TGLAxis.h"

#include "TGLRnrCtx.h"

#include "TGLIncludes.h"

//______________________________________________________________________
// TH2GL
//
// Rendering of TH2 and derived classes.
// Interface to plot-painters also used for gl-in-pad.

ClassImp(TH2GL)

TH2GL::TH2GL() : TGLObject(), fM(0)
{
   // Constructor.

   fDLCache = kFALSE; // Disable display list.
}

TH2GL::~TH2GL()
{
   // Destructor.

   delete fPlotPainter;
}

/**************************************************************************/

Bool_t TH2GL::SetModel(TObject* obj, const Option_t* opt)
{
   // Set model object.

   if(SetModelCheckClass(obj, TH2::Class()))
   {
      fM = dynamic_cast<TH2*>(obj);

      TString option(opt);

      // Plot type
      if (option.Index("iso") != kNPOS)
         fPlotPainter = new TGLIsoPainter(fM, 0, &fCoord);
      else if (option.Index("box") != kNPOS)
         fPlotPainter = new TGLBoxPainter(fM, 0, &fCoord);
      // else if (option.Index("tf3") != kNPOS)
      //    fPlotPainter = new TGLTF3Painter(fF3, fM, 0, &fCoord);
      else if (option.Index("surf") != kNPOS)
         fPlotPainter = new TGLSurfacePainter(fM, 0, &fCoord);
      else
         fPlotPainter = new TGLLegoPainter(fM, 0, &fCoord);
   
      // Coord-system
      fCoord.SetXLog(gPad->GetLogx());
      fCoord.SetYLog(gPad->GetLogy());
      fCoord.SetZLog(gPad->GetLogz());

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

void TH2GL::SetBBox()
{
   // Setup bounding-box.

   fBoundingBox.Set(fPlotPainter->RefBackBox().Get3DBox());
}

/**************************************************************************/

void TH2GL::DirectDraw(TGLRnrCtx & /*rnrCtx*/) const
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
