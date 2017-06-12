// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveSceneInfo.h"
#include "TEveScene.h"
#include "TEveManager.h"

#include "TGLSceneInfo.h"

/** \class TEveSceneInfo
\ingroup TEve
TEveUtil representation of TGLSceneInfo.
*/

ClassImp(TEveSceneInfo);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveSceneInfo::TEveSceneInfo(TEveViewer* viewer, TEveScene* scene, TGLSceneInfo* sinfo) :
   TEveElement (),
   TNamed        (Form("SI - %s", scene->GetName()),
                  Form("TEveSceneInfo of scene '%s'", scene->GetName())),
   fViewer       (viewer),
   fScene        (scene),
   fGLSceneInfo  (sinfo)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Return the TGLSceneBase represented by this SceneInfo object.

TGLSceneBase* TEveSceneInfo::GetGLScene() const
{
   return fGLSceneInfo->GetScene();
}

////////////////////////////////////////////////////////////////////////////////
/// Override from TEveElement.
/// Process visibility changes and forward them to fGLScene.

void TEveSceneInfo::AddStamp(UChar_t bits)
{
   TEveElement::AddStamp(bits);
   if (bits & kCBVisibility)
   {
      fGLSceneInfo->SetActive(fRnrSelf);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.
/// TEveSceneInfo does not accept children.

Bool_t TEveSceneInfo::AcceptElement(TEveElement* /*el*/)
{
   static const TEveException eH("TEveSceneInfo::AcceptElement ");

   gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.
/// TEveSceneInfo does not accept children.

Bool_t TEveSceneInfo::HandleElementPaste(TEveElement* /*el*/)
{
   static const TEveException eH("TEveSceneInfo::HandleElementPaste ");

   gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}
