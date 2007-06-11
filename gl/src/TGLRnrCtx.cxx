// @(#)root/gl:$Name$:$Id$
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
//
// !!! remove DL handling ... should go to gl-context.


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

   fQuadric (0),

   fDLCaptureOpen (kFALSE)
{
   // Constructor.

   fSelectBuffer = new TGLSelectBuffer;
}

//______________________________________________________________________
TGLRnrCtx::~TGLRnrCtx()
{
   // Destructor.

   delete fPickRectangle;
   delete fSelectBuffer;
   // !!! Should 'cd into GL-context' to destroy display-lists and quadric.
   if (!gVirtualX->IsCmdThread()) {
      gROOT->ProcessLineFast(Form("((TGLRnrCtx *)0x%x)->DestroyQuadric()", this));
      gROOT->ProcessLineFast(Form("((TGLRnrCtx *)0x%x)->ProcessDLWipeList()", this));
   } else {
      DestroyQuadric();
      ProcessDLWipeList();
   }
}

//______________________________________________________________________
void TGLRnrCtx::DestroyQuadric()
{
   // Destroy the shared quadric.

   if (fQuadric) gluDeleteQuadric(fQuadric);
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

//______________________________________________________________________
void TGLRnrCtx::Reset()
{
   // Reset the context. No longer needed, done in Viewer/ViewerBase.
   //
   // This also initializes GL interface stuff (display-list cache,
   // quadric object) and so must be called from within a valid
   // GL-context.
   // !!! Will be removed when Timur provides hooks in TGLContext.

   if (!fQuadric)
   {
      fQuadric = gluNewQuadric();
      if (!fQuadric) {
         Error("TGLRnrCtx::Reset", "create quadric failed.");
      } else {
         gluQuadricOrientation(fQuadric, (GLenum)GLU_OUTSIDE);
         gluQuadricNormals    (fQuadric, (GLenum)GLU_SMOOTH);
      }
   }
   ProcessDLWipeList();
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
void TGLRnrCtx::ProcessDLWipeList()
{
   // Unregister name-ranges that are no longer needed from Gl.

   while (!fDLWipeList.empty())
   {
      DLNameRange_t & rng = fDLWipeList.front();
      glDeleteLists(rng.first, rng.second);
      fDLWipeList.pop_front();
   }
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

//______________________________________________________________________
void TGLRnrCtx::RegisterDLNameRangeToWipe(UInt_t base, Int_t size)
{
   // Register a rande of display-list names to be wiped
   // once the context becomes current.

   fDLWipeList.push_back(DLNameRange_t(base, size));
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
