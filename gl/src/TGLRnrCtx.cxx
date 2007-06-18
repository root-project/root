// @(#)root/gl:$Name:  $:$Id: TGLRnrCtx.cxx,v 1.1 2007/06/11 19:56:33 brun Exp $
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TVirtualX.h"
#include "TString.h"
#include "TROOT.h"

#include "TGLRnrCtx.h"
#include "TGLSceneInfo.h"
#include "TGLSelectBuffer.h"
#include "TGLIncludes.h"
#include "TGLUtil.h"

#include <TError.h>
#include <TMathBase.h>

#include <algorithm>
#include <assert.h>

//______________________________________________________________________
// TGLRnrCtx
//
// The TGLRnrCtx class aggregates data for a given redering context as
// needed by various parts of the ROOT's OpenGL infractructure. It
// serves as a connecting point between the steering part of the
// infrastructure (viewer, scene) and concrete rendering classes
// (logical, physical shape). It is just a data-holder, there is no
// functionality in it.
//
// Development notes:
//
// One RnrCtx is created by each viewer and it is just an extension of
// the viewer context that changes along the render
// descend. Separating this also has some abstract benefit of hiding
// viewer implementation from those that do not need to know it.
//
// Current scene / scene-info part is always initialized by viewer,
// scenes can assume they're ok.


ClassImp(TGLRnrCtx)

//______________________________________________________________________
TGLRnrCtx::TGLRnrCtx(TGLViewerBase* viewer) :
   fViewer    (viewer),
   fCamera    (0),
   fSceneInfo (0),

   fViewerLOD    (kLODUndef),
   fSceneLOD     (kLODUndef),
   fShapeLOD     (kLODUndef),

   fViewerStyle  (kStyleUndef),
   fSceneStyle   (kStyleUndef),

   fViewerClip   (0),
   fSceneClip    (0),
   fClip         (0),
   fDrawPass     (kPassUndef),

   fRenderTimeout(0.0),

   fSelection    (kFALSE),
   fSecSelection (kFALSE),
   fPickRectangle(0),
   fSelectBuffer (0),

   fDLCaptureOpen (kFALSE),
   fGLCtxIdentity (0),
   fQuadric       (0)
{
   // Constructor.

   fSelectBuffer = new TGLSelectBuffer;
   fQuadric = gluNewQuadric();
   gluQuadricOrientation(fQuadric, (GLenum)GLU_OUTSIDE);
   gluQuadricNormals    (fQuadric, (GLenum)GLU_SMOOTH);
}

//______________________________________________________________________
TGLRnrCtx::~TGLRnrCtx()
{
   // Destructor.

   gluDeleteQuadric(fQuadric);
   delete fPickRectangle;
   delete fSelectBuffer;
}

//______________________________________________________________________
TGLSceneBase * TGLRnrCtx::GetScene()
{
   // Return current scene (based on scene-info data).

   return  fSceneInfo->GetScene();
}

//______________________________________________________________________
TGLSceneBase & TGLRnrCtx::RefScene()
{
   // Return current scene (based on scene-info data).

   return *fSceneInfo->GetScene();
}

/**************************************************************************/

void TGLRnrCtx::BeginSelection(Int_t x, Int_t y, Int_t r)
{
   // Setup context for running selection.
   // x and y are in window coordinates.

   fSelection    = kTRUE;
   fSecSelection = kFALSE;
   if (!fPickRectangle) fPickRectangle = new TGLRect;
   fPickRectangle->Set(x, y, r, r);

   glSelectBuffer(fSelectBuffer->GetBufSize(), fSelectBuffer->GetBuf());
}

void TGLRnrCtx::EndSelection(Int_t glResult)
{
   // End selection.

   fSelection    = kFALSE;
   fSecSelection = kFALSE;
   delete fPickRectangle; fPickRectangle = 0;

   if (glResult < 0)
   {
      if (fSelectBuffer->CanGrow())
      {
         Warning("TGLRnrCtx::EndSelection",
                 "Select buffer size (%d) insufficient, doubling it.",
                 fSelectBuffer->GetBufSize());
         fSelectBuffer->Grow();
      }
      else
      {
         Warning("TGLRnrCtx::EndSelection",
                 "Select buffer size (%d) insufficient. This is maximum.",
                 fSelectBuffer->GetBufSize());
      }
   }
   fSelectBuffer->ProcessResult(glResult);
}

TGLRect * TGLRnrCtx::GetPickRectangle()
{
   // Return current pick rectangle. This is *zero* when
   // selection is not set.

   return fPickRectangle;
}

Int_t TGLRnrCtx::GetPickRadius()
{
   // Return pick radius. If selection is not set it returns the
   // default vale.

   return fPickRectangle ? fPickRectangle->Width() : 3;
}

/**************************************************************************/

//______________________________________________________________________
Bool_t TGLRnrCtx::IsDrawPassFilled() const
{
   // Returns true if current render-pass uses filled polygon style.

   return fDrawPass == kPassFill || fDrawPass == kPassOutlineFill;
}

//______________________________________________________________________
void TGLRnrCtx::OpenDLCapture()
{
   // Start display-list capture.

   assert(fDLCaptureOpen == kFALSE);
   fDLCaptureOpen = kTRUE;
}

//______________________________________________________________________
void TGLRnrCtx::CloseDLCapture()
{
   // End display list capture.

   assert(fDLCaptureOpen == kTRUE);
   fDLCaptureOpen = kFALSE;
}


/**************************************************************************/
// Static helpers
/**************************************************************************/

//______________________________________________________________________________
const char* TGLRnrCtx::StyleName(Short_t style)
{
   // Return string describing the style.

   switch (style)
   {
      case TGLRnrCtx::kFill:       return "Filled Polys";
      case TGLRnrCtx::kWireFrame:  return "Wireframe";
      case TGLRnrCtx::kOutline:    return "Outline";
      default:                     return "Oogaa-dooga style";
   }
}
