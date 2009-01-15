// @(#)root/gl:$Id$
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
#include "TGLFontManager.h"
#include "TGLContext.h"

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
   fCombiLOD     (kLODUndef),
   fShapeLOD     (kLODUndef),

   fViewerStyle  (kStyleUndef),
   fSceneStyle   (kStyleUndef),

   fViewerClip   (0),
   fSceneClip    (0),
   fClip         (0),
   fDrawPass     (kPassUndef),

   fStopwatch    (),
   fRenderTimeOut(0.0),
   fIsRunning    (kFALSE),
   fHasTimedOut  (kFALSE),

   fHighlight    (kFALSE),  fHighlightOutline (kFALSE),
   fSelection    (kFALSE),  fSecSelection     (kFALSE),
   fPickRadius   (0),
   fPickRectangle(0),
   fSelectBuffer (0),

   fDLCaptureOpen (kFALSE),
   fGLCtxIdentity (0),
   fQuadric       (0),

   fGrabImage     (kFALSE),
   fGrabbedImage  (0)
{
   // Constructor.

   fSelectBuffer = new TGLSelectBuffer;
   fQuadric = gluNewQuadric();
   gluQuadricOrientation(fQuadric, (GLenum)GLU_OUTSIDE);
   gluQuadricNormals    (fQuadric, (GLenum)GLU_SMOOTH);

   if (fViewer == 0)
   {
      // Assume external usage, initialize for highest quality.
      fViewerLOD = fSceneLOD = fCombiLOD = fShapeLOD = kLODHigh;
      fViewerStyle = fSceneStyle = kFill;
      fDrawPass = kPassFill;
   }

   // Colors for different shape-selection-levels.
   fSSLColor[0][0] =   0; fSSLColor[0][1] =   0; fSSLColor[0][2] =   0; fSSLColor[0][3] =   0;
   fSSLColor[1][0] = 255; fSSLColor[1][1] = 255; fSSLColor[1][2] = 255; fSSLColor[1][3] = 255;
   fSSLColor[2][0] = 255; fSSLColor[2][1] = 255; fSSLColor[2][2] = 255; fSSLColor[2][3] = 255;
   fSSLColor[3][0] = 200; fSSLColor[3][1] = 200; fSSLColor[3][2] = 255; fSSLColor[3][3] = 255;
   fSSLColor[4][0] = 200; fSSLColor[4][1] = 200; fSSLColor[4][2] = 255; fSSLColor[4][3] = 255;
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

//______________________________________________________________________
Bool_t TGLRnrCtx::IsDrawPassFilled() const
{
   // Returns true if current render-pass uses filled polygon style.

   return fDrawPass == kPassFill || fDrawPass == kPassOutlineFill;
}


/******************************************************************************/
// Stopwatch
/******************************************************************************/

//______________________________________________________________________________
void TGLRnrCtx:: StartStopwatch()
{
   // Start the stopwatch.

   if (fIsRunning)
      return;

   fStopwatch.Start();
   fIsRunning   = kTRUE;
   fHasTimedOut = kFALSE;
}

//______________________________________________________________________________
void TGLRnrCtx:: StopStopwatch()
{
   // Stop the stopwatch.

   fHasTimedOut = fStopwatch.End() > fRenderTimeOut;
   fIsRunning = kFALSE;
}

//______________________________________________________________________________
Bool_t TGLRnrCtx::HasStopwatchTimedOut()
{
   // Check if the stopwatch went beyond the render time limit.

   if (fHasTimedOut) return kTRUE;
   if (fIsRunning && fStopwatch.Lap() > fRenderTimeOut)
      fHasTimedOut = kTRUE;
   return fHasTimedOut;
}


/******************************************************************************/
// Selection & picking
/******************************************************************************/

//______________________________________________________________________________
void TGLRnrCtx::BeginSelection(Int_t x, Int_t y, Int_t r)
{
   // Setup context for running selection.
   // x and y are in window coordinates.

   fSelection    = kTRUE;
   fSecSelection = kFALSE;
   fPickRadius   = r;
   if (!fPickRectangle) fPickRectangle = new TGLRect;
   fPickRectangle->Set(x, y, r, r);

   glSelectBuffer(fSelectBuffer->GetBufSize(), fSelectBuffer->GetBuf());
}

//______________________________________________________________________________
void TGLRnrCtx::EndSelection(Int_t glResult)
{
   // End selection.

   fSelection    = kFALSE;
   fSecSelection = kFALSE;
   fPickRadius   = 0;
   delete fPickRectangle; fPickRectangle = 0;

   if (glResult < 0)
   {
      if (fSelectBuffer->CanGrow() && fSelectBuffer->GetBufSize() > 0x10000)
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

//______________________________________________________________________________
TGLRect * TGLRnrCtx::GetPickRectangle()
{
   // Return current pick rectangle. This is *zero* when
   // selection is not set.

   return fPickRectangle;
}

//______________________________________________________________________________
Int_t TGLRnrCtx::GetPickRadius()
{
   // Return pick radius. If selection is not active it returns 0.

   return fPickRadius;
}


/******************************************************************************/
// Colors for Shape-Selection-Level
/******************************************************************************/

void TGLRnrCtx::SetSSLColor(Int_t level, UChar_t r, UChar_t g, UChar_t b, UChar_t a)
{
   // Set highlight color for shape-selection-level level.

   fSSLColor[level][0] = r;
   fSSLColor[level][1] = g;
   fSSLColor[level][2] = b;
   fSSLColor[level][3] = a;
}

void TGLRnrCtx::SetSSLColor(Int_t level, UChar_t rgba[4])
{
   // Set highlight color for shape-selection-level level.

   fSSLColor[level][0] = rgba[0];
   fSSLColor[level][1] = rgba[1];
   fSSLColor[level][2] = rgba[2];
   fSSLColor[level][3] = rgba[3];
}


/**************************************************************************/
// Display-list state
/******************************************************************************/

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


/******************************************************************************/
// TGLFont interface
/******************************************************************************/

//______________________________________________________________________
void TGLRnrCtx::RegisterFont(Int_t size, Int_t file, Int_t mode, TGLFont& out)
{
   // Get font in the GL rendering context.

   fGLCtxIdentity->GetFontManager()->RegisterFont(size, file, (TGLFont::EMode)mode, out);
}

//______________________________________________________________________
void TGLRnrCtx::RegisterFont(Int_t size, const char* name, Int_t mode, TGLFont& out)
{
   // Get font in the GL rendering context.

   fGLCtxIdentity->GetFontManager()->RegisterFont(size, name, (TGLFont::EMode)mode, out);
}
//______________________________________________________________________
void TGLRnrCtx::ReleaseFont(TGLFont& font)
{
   // Release font in the GL rendering context.

   fGLCtxIdentity->GetFontManager()->ReleaseFont(font);
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
