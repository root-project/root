// @(#)root/gl:$Id$
// Author:  Matevz Tadel, Feb 2007

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGLSceneInfo.h"
#include "TGLRnrCtx.h"
#include "TGLSceneBase.h"
#include "TGLViewerBase.h"

/** \class TGLSceneInfo
\ingroup opengl
Base class for extended scene context.

Scenes can be shared among several viewers and each scene needs to
cache some viewer/camera/clipping specific state => this is a
storage class for this data.

Sub-classes of TGLSceneBase can override the virtual
CreateSceneInfo() method and in it instantiate a sub-class of
TGLSceneInfo containing the needed information. See TGLScene and
inner class SceneInfo; note that some casting is needed in actual
methods as TGLRnrCtx holds the base-class pointer.
*/

ClassImp(TGLSceneInfo);

////////////////////////////////////////////////////////////////////////////////

TGLSceneInfo::TGLSceneInfo(TGLViewerBase* view, TGLSceneBase* scene) :
   fViewer    (view),
   fScene     (scene),
   fActive    (kTRUE),

   fLOD     (TGLRnrCtx::kLODUndef),
   fStyle   (TGLRnrCtx::kStyleUndef),
   fWFLineW (0),
   fOLLineW (0),
   fClip    (0),

   fLastLOD   (TGLRnrCtx::kLODUndef),
   fLastStyle (TGLRnrCtx::kStyleUndef),
   fLastWFLineW (0),
   fLastOLLineW (0),
   fLastClip  (0),
   fLastCamera(0),

   fSceneStamp (0),
   fClipStamp  (0),
   fCameraStamp(0),
   fUpdateTimeouted(kFALSE),

   fViewCheck (kTRUE),
   fInFrustum (kTRUE),
   fInClip    (kTRUE),
   fClipMode  (kClipNone)
{
   // Default constructor.
}

////////////////////////////////////////////////////////////////////////////////
/// Set active state of the scene, mark viewer as changed.

void TGLSceneInfo::SetActive(Bool_t a)
{
   if (a != fActive)
   {
      fActive = a;
      fViewer->Changed();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Combine information from scene, scene-info and camera (should be
/// optional) into transformation matrices.
///
/// Transform scene bounding box using this transformation.

void TGLSceneInfo::SetupTransformsAndBBox()
{
   // !!! Transforms not implemented yet, just copy the scene bounding
   // box.

   fTransformedBBox = fScene->BoundingBox();
}
