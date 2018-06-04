// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TEveSceneInfo.hxx"
#include "ROOT/TEveScene.hxx"

using namespace ROOT::Experimental;
namespace REX = ROOT::Experimental;

/** \class TEveSceneInfo
\ingroup TEve
Representation of a TEveScene in a TEveViewer. This allows for
viewer specific settings to be applied to a scene, e.g., global position
in viewer coordinate system.
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveSceneInfo::TEveSceneInfo(TEveViewer* viewer, TEveScene* scene) :
   TEveElement (),
   TNamed        (Form("SI - %s", scene->GetName()),
                  Form("TEveSceneInfo of scene '%s'", scene->GetName())),
   fViewer       (viewer),
   fScene        (scene)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t TEveSceneInfo::WriteCoreJson(nlohmann::json& j, Int_t rnr_offset)
{
   Int_t ret = TEveElement::WriteCoreJson(j, rnr_offset);

   j["fSceneId"] = fScene->GetElementId();

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Override from TEveElement.
/// Process visibility changes and forward them to fGLScene.

void TEveSceneInfo::AddStamp(UChar_t bits)
{
   TEveElement::AddStamp(bits);
   // if (bits & kCBVisibility)
   // {
   //    fGLSceneInfo->SetActive(fRnrSelf);
   // }
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.
/// TEveSceneInfo does not accept children.

Bool_t TEveSceneInfo::AcceptElement(TEveElement* /*el*/)
{
   static const TEveException eH("TEveSceneInfo::AcceptElement ");

   // gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual from TEveElement.
/// TEveSceneInfo does not accept children.

Bool_t TEveSceneInfo::HandleElementPaste(TEveElement* /*el*/)
{
   static const TEveException eH("TEveSceneInfo::HandleElementPaste ");

   // gEve->SetStatusLine(eH + "this class does not accept children.");
   return kFALSE;
}
