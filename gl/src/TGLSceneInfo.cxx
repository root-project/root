// @(#)root/gl:$Name$:$Id$
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

//______________________________________________________________________
// TGLSceneInfo
//
// Base class for extended scene context.
//
// Scenes can be shared among several viewers and each scene needs to
// cache some viewer/camera/clipping specific state => this is a
// storage class for this data.
//
// Sub-classes of TGLSceneBase can override the virtual
// CreateSceneInfo() method and in it instantiate a sub-class of
// TGLSceneInfo containing the needed information. See TGLScene and
// inner class SceneInfo; note that some casting is needed in actual
// methods as TGLRnrCtx holds the base-class pointer.
//

ClassImp(TGLSceneInfo)


//______________________________________________________________________
TGLSceneInfo::TGLSceneInfo(TGLViewerBase* view, TGLSceneBase* scene) :
   fViewer    (view),
   fScene     (scene),

   fLOD   (TGLRnrCtx::kLODUndef),
   fStyle (TGLRnrCtx::kStyleUndef),
   fClip  (0),

   fLastLOD   (TGLRnrCtx::kLODUndef),
   fLastStyle (TGLRnrCtx::kStyleUndef),
   fLastClip  (0),
   fLastCamera(0),

   fSceneStamp (0),
   fClipStamp  (0),
   fCameraStamp(0),

   fViewCheck (kTRUE),
   fInFrustum (kTRUE),
   fInClip    (kTRUE),
   fClipMode  (kClipNone)
{
   // Default constructor.
}

//______________________________________________________________________
void TGLSceneInfo::SetupTransformsAndBBox()
{
   // Combine information from scene, scene-info and camera(?) into
   // transformation matrices.
   // Transform scene bounding box using this transformation.

   // !!! Transforms not implemented yet, just copy the scene bounding
   // box.

   fTransformedBBox = fScene->BoundingBox();
}
