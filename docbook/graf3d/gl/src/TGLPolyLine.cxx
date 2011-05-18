// @(#)root/gl:$Id$
// Author:  Timur Pocheptsov  03/08/2004
// NOTE: This code moved from obsoleted TGLSceneObject.h / .cxx - see these
// attic files for previous CVS history

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include "TGLPolyLine.h"
#include "TGLRnrCtx.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TAttLine.h"

// For debug tracing
#include "TClass.h"
#include "TError.h"

//______________________________________________________________________________
/* Begin_Html
<center><h2>GL Polyline</h2></center>
To draw a 3D polyline in a GL window.
End_Html */

ClassImp(TGLPolyLine)

//______________________________________________________________________________
TGLPolyLine::TGLPolyLine(const TBuffer3D & buffer) :
   TGLLogicalShape(buffer),
   fVertices(buffer.fPnts, buffer.fPnts + 3 * buffer.NbPnts()),
   fLineWidth(1.)
{
   // constructor
   //dynamic_cast because of multiple inheritance.
   if (TAttLine *lineAtt = dynamic_cast<TAttLine *>(buffer.fID))
      fLineWidth = lineAtt->GetLineWidth();
}


//______________________________________________________________________________
void TGLPolyLine::DirectDraw(TGLRnrCtx & rnrCtx) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyLine::DirectDraw", "this %ld (class %s) LOD %d", (Long_t)this, IsA()->GetName(), rnrCtx.ShapeLOD());
   }

   if (rnrCtx.DrawPass() == TGLRnrCtx::kPassOutlineLine)
      return;

   Double_t oldWidth = 1.;
   glGetDoublev(GL_LINE_WIDTH, &oldWidth);

   TGLUtil::LineWidth(fLineWidth);

   glBegin(GL_LINE_STRIP);

   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);

   glEnd();

   glLineWidth(oldWidth);
}
