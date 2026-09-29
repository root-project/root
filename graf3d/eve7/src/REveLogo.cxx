// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/REveLogo.hxx>
#include <ROOT/REveManager.hxx>
#include <ROOT/REveRenderData.hxx>

#include "TSystem.h"

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;

/** \class REveLogo
\ingroup REve
A screen-space image for an overlay scene, such as an experiment logo or a
watermark.

The client draws it as a RenderCore ZLogo, a screen-space ZSprite. The size is
in CSS pixels and does not change with the camera, and the outline comes from
the image alpha. The logo can be moved, and resized from its corner grip, in
the client only. Nothing is sent back, so each client places its own copy.

Images are served from a directory registered with SetImageDir() and mapped
under "eve-images/", as REveText maps its SDF fonts. ROOT does not ship any
images.
*/

std::string REveLogo::sImageDir;

////////////////////////////////////////////////////////////////////////////////
/// Constructor. A `file` that IsRemote() accepts selects kRemote.

REveLogo::REveLogo(std::string_view file, const Text_t *n, const Text_t *t) : REveElement(n, t), fFile(file)
{
   SetPickable(kTRUE); // otherwise it cannot be moved

   if (IsRemote(fFile))
      fSource = kRemote;
}

////////////////////////////////////////////////////////////////////////////////
/// Is the file an absolute URL, to be fetched by the client directly?

bool REveLogo::IsRemote(const std::string &f)
{
   return f.compare(0, 7, "http://") == 0 || f.compare(0, 8, "https://") == 0 ||
          f.compare(0, 2, "//") == 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Register the directory images are served from, mapped under "eve-images/".
/// Mirrors REveText::SetSdfFontDir; must run after REveManager::Create().

bool REveLogo::SetImageDir(std::string_view dir)
{
   static const char *tpfx = "REveLogo::SetImageDir";

   if (!gEve) {
      ::Error(tpfx, "REveManager needs to be initialized before the image dir can be set.");
      return false;
   }

   std::string sanitized(dir);
   if (sanitized.empty()) {
      ::Error(tpfx, "Empty directory given.");
      return false;
   }
   if (sanitized.back() != '/')
      sanitized += '/';

   if (gSystem->AccessPathName(sanitized.c_str(), kReadPermission)) {
      ::Error(tpfx, "Image directory '%s' does not exist or is not readable.", sanitized.c_str());
      return false;
   }

   sImageDir = sanitized;
   gEve->AddLocation("eve-images/", sImageDir);
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Stream the file, source, position, size, opacity and resizability.

Int_t REveLogo::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveElement::WriteCoreJson(j, rnr_offset);

   j["fFile"] = fFile;
   j["fSource"] = (int)fSource;
   j["fPosX"] = fPosX;
   j["fPosY"] = fPosY;
   j["fSize"] = fSize;
   j["fOpacity"] = fOpacity;
   j["fResizable"] = fResizable;

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Name the client factory makeLogo. No geometry is sent.

void REveLogo::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeLogo");
   REveElement::BuildRenderData();
   fRenderData->PushV(0.f, 0.f, 0.f); // keep the buffer non-empty
}
