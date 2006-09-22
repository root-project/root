// @(#)root/gl:$Name:  $:$Id: TGLPolyLine.cxx,v 1.2 2006/05/31 07:48:56 brun Exp $
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
#include "TGLDrawFlags.h"
#include "TGLIncludes.h"

#include "TBuffer3D.h"
#include "TBuffer3DTypes.h"
#include "TAttLine.h"

// For debug tracing
#include "TClass.h" 
#include "TError.h"

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
void TGLPolyLine::DirectDraw(const TGLDrawFlags & flags) const
{
   // Debug tracing
   if (gDebug > 4) {
      Info("TGLPolyLine::DirectDraw", "this %d (class %s) LOD %d", this, IsA()->GetName(), flags.LOD());
   }

   Double_t oldWidth = 1.;
   glGetDoublev(GL_LINE_WIDTH, &oldWidth);
   
   glLineWidth(fLineWidth);

   glBegin(GL_LINE_STRIP);

   for (UInt_t i = 0; i < fVertices.size(); i += 3)
      glVertex3d(fVertices[i], fVertices[i + 1], fVertices[i + 2]);

   glEnd();

   glLineWidth(oldWidth);
}
