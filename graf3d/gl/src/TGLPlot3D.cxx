// @(#)root/gl:$Id$
// Author: Matevz Tadel 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLPlot3D.h"

#include "TH3.h"
#include "TH3GL.h"
#include "TH2.h"
#include "TH2GL.h"
#include "TF2.h"
#include "TF2GL.h"
#include "TGLParametric.h"
#include "TPolyMarker3D.h"
#include "TGLParametricEquationGL.h"

#include "TVirtualPad.h"

/** \class TGLPlot3D
\ingroup opengl
Description of TGLPlot3D
*/

ClassImp(TGLPlot3D);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TGLPlot3D::TGLPlot3D() : TGLObject(), fPlotPainter(0)
{
   fDLCache = kFALSE; // Disable display list.
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TGLPlot3D::~TGLPlot3D()
{
   delete fPlotPainter;
}

////////////////////////////////////////////////////////////////////////////////
/// Set painter object and destroy the old one.

void TGLPlot3D::SetPainter(TGLPlotPainter* p)
{
   delete fPlotPainter;
   fPlotPainter = p;
}

////////////////////////////////////////////////////////////////////////////////
/// Instantiate the correct plot-painter for given object.
/// Protected method.

TGLPlot3D* TGLPlot3D::InstantiatePlot(TObject* obj)
{
   if (obj->InheritsFrom(TH3::Class()))
   {
      return new TH3GL();
   }
   else if (obj->InheritsFrom(TH2::Class()))
   {
      return new TH2GL();
   }
   else if (obj->InheritsFrom(TF2::Class()))
   {
      return new TF2GL();
   }
   else if (obj->InheritsFrom(TGLParametricEquation::Class()))
   {
      return new TGLParametricEquationGL();
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Create GL plot for specified TH3 and polymarker.

TGLPlot3D* TGLPlot3D::CreatePlot(TH3 *th3, TPolyMarker3D *pm)
{
   TGLPlot3D* log = new TH3GL(th3, pm);
   log->SetBBox();

   return log;
}

////////////////////////////////////////////////////////////////////////////////
/// Create GL plot for specified object and options.
/// Which axes are logarithmic is determined from a pad.

TGLPlot3D* TGLPlot3D::CreatePlot(TObject* obj, const Option_t* opt, TVirtualPad* pad)
{
   TGLPlot3D* log = InstantiatePlot(obj);

   if (log)
   {
      log->fCoord.SetXLog(pad->GetLogx());
      log->fCoord.SetYLog(pad->GetLogy());
      log->fCoord.SetZLog(pad->GetLogz());
      log->SetModel(obj, opt);
      log->SetBBox();
   }

   return log;
}

////////////////////////////////////////////////////////////////////////////////
/// Create GL plot for specified object and options.
/// Which axes are logarithmic is determined from explicit arguments.

TGLPlot3D* TGLPlot3D::CreatePlot(TObject* obj, const Option_t* opt, Bool_t logx, Bool_t logy, Bool_t logz)
{
   TGLPlot3D* log = InstantiatePlot(obj);

   if (log)
   {
      log->fCoord.SetXLog(logx);
      log->fCoord.SetYLog(logy);
      log->fCoord.SetZLog(logz);
      log->SetModel(obj, opt);
      log->SetBBox();
   }

   return log;
}
