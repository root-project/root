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

//______________________________________________________________________________
// TEveSceneInfo
//
// TEveUtil representation of TGLSceneInfo.

ClassImp(TEveSceneInfo)

//______________________________________________________________________________
TEveSceneInfo::TEveSceneInfo(TEveViewer* viewer, TEveScene* scene, TGLSceneInfo* sinfo) :
   TEveElement (),
   TNamed        (Form("SI - %s", scene->GetName()),
                  Form("TEveSceneInfo of scene '%s'", scene->GetName())),
   fViewer       (viewer),
   fScene        (scene),
   fGLSceneInfo  (sinfo)
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
TGLSceneBase* TEveSceneInfo::GetGLScene() const
{
   // Return the TGLSceneBase represented by this SceneInfo object.

   return fGLSceneInfo->GetScene();
}

/******************************************************************************/

//______________________________________________________________________________
void TEveSceneInfo::AddStamp(UChar_t bits)
{
   // Override from TEveElement.
   // Process visibility changes and forward them to fGLScene.

   TEveElement::AddStamp(bits);
   if (bits & kCBVisibility)
   {
      fGLSceneInfo->SetActive(fRnrSelf);
   }
}

/******************************************************************************/

//______________________________________________________________________________
Bool_t TEveSceneInfo::AcceptElement(TEveElement* /*el*/)
{
   // Virtual from TEveElement.
   // TEveSceneInfo does not accept children.

   static const TEveException eH("TEveSceneInfo::AcceptElement ");

   gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TEveSceneInfo::HandleElementPaste(TEveElement* /*el*/)
{
   // Virtual from TEveElement.
   // TEveSceneInfo does not accept children.

   static const TEveException eH("TEveSceneInfo::HandleElementPaste ");

   gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}
