// @(#)root/eve7:$Id$
// Author: Waad Alshehri, 2023

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <ROOT/REveText.hxx>
#include <ROOT/REveRenderData.hxx>

#include <nlohmann/json.hpp>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

REveText::REveText(const Text_t* n, const Text_t* t) :
   REveShape(n, t)
{
   // MainColor set to FillColor in Shape.
   fPickable  = true;
   fLineWidth = 0.05; // override, in text-size units
}

////////////////////////////////////////////////////////////////////////////////
/// Fill core part of JSON representation.

Int_t REveText::WriteCoreJson(nlohmann::json &j, Int_t rnr_offset)
{
   Int_t ret = REveShape::WriteCoreJson(j, rnr_offset);

   j["fText"] = fText;
   j["fFont"] = fFont;
   j["fPosX"] = fPosition.fX;
   j["fPosY"] = fPosition.fY;
   j["fPosZ"] = fPosition.fZ;
   j["fFontSize"] = fFontSize;
   j["fFontHinting"] = fFontHinting;
   j["fExtraBorder"] = fExtraBorder;
   j["fMode"] = fMode;
   j["fTextColor"] = fTextColor;

   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Crates 3D point array for rendering.

void REveText::BuildRenderData()
{
   fRenderData = std::make_unique<REveRenderData>("makeZText");
   REveShape::BuildRenderData();
   // TODO write fPosition and fFontSize here ...
   fRenderData->PushV(0.f, 0.f, 0.f); // write floats so the data is not empty
}

////////////////////////////////////////////////////////////////////////////////
/// Compute bounding-box of the data.

void REveText::ComputeBBox()
{
   //BBoxInit();
}


#include "ROOT/REveManager.hxx"
#include "TSystem.h"
#include "TROOT.h"
#include "TEnv.h"

std::string REveText::sSdfFontDir;

////////////////////////////////////////////////////////////////////////////////
/// Set location where SDF fonts and their metrics data are stored or are to be
/// created via the AssertSdfFont() static function.
/// If require_write_access is true (default), write permission in the directory
//  dir is required.
/// REveManager needs to be created before calling this function.
/// Static function.

bool REveText::SetSdfFontDir(std::string_view dir, bool require_write_access)
{
   static const char* tpfx = "REveText::SetSdfFontDir";

   if (gEve == nullptr) {
      ::Error(tpfx, "REveManager needs to be initialized before font setup can begin.");
      return false;
   }

   std::string sanitized_dir(dir);
   if (sanitized_dir.back() != '/')
      sanitized_dir += '/';
   if (gSystem->AccessPathName(sanitized_dir.data())) {
      if (gSystem->mkdir(sanitized_dir.data(), true)) {
         ::Error(tpfx, "Directory does not exist and mkdir failed for '%s", dir.data());
         return false;
      }
   }
   auto dir_perms = require_write_access ? kWritePermission : kReadPermission;
   if (gSystem->AccessPathName(sanitized_dir.data(), dir_perms) == false) {
      sSdfFontDir = sanitized_dir;
      gEve->AddLocation("sdf-fonts/", sSdfFontDir.data());
      return true;
   } else {
      return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set default SDF font directory based on write permissions in $ROOTSYS and
/// in the current working directory.
/// Alternative fallback to /tmp or user's home directory is not attempted.

bool REveText::SetDefaultSdfFontDir()
{
   static const char* tpfx = "REveText::SetDefaultSdfFontDir";

   static bool s_font_init_failed = false;

   if (s_font_init_failed) {
      return false;
   }

   std::string dir( gEnv->GetValue("WebGui.RootUi5Path", gSystem->ExpandPathName("${ROOTSYS}/ui5")) );
   s_font_init_failed = true;
   if (SetSdfFontDir(dir + "/eve7/sdf-fonts/")) {
      ::Info(tpfx, "Using install-wide SDF font dir $ROOTSYS/ui5/eve7/sdf-fonts");
   } else if (SetSdfFontDir("./sdf-fonts/")) {
      ::Info(tpfx, "Using SDF font dir sdf_fonts/ in current directory");
   } else {
      ::Error(tpfx, "Error setting up default SDF font dir. "
                    "Please set it manually through REveText::SetSdfFontDir(<dir-name>)");
      return false;
   }
   s_font_init_failed = false;

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if font exists, otherwise try to create it.
/// If SDF font dir is not yet set, an attempt will be made to set it to
/// one of the default locations, in $ROOTSYS or in the current working directory.
/// Returns true if font files are present, false otherwise.
/// Static function.

bool REveText::AssertSdfFont(std::string_view font_name, std::string_view ttf_font)
{
   static const char* tpfx = "REveText::AssertSdfFont";

   if (sSdfFontDir.empty() && ! SetDefaultSdfFontDir()) {
      return false;
   }

   std::string base = sSdfFontDir + font_name.data();
   std::string png = base + ".png";
   std::string js  = base + ".js.gz";

   if (gSystem->AccessPathName(png.data()) || gSystem->AccessPathName(js.data())) {
      if (gSystem->AccessPathName(ttf_font.data())) {
         ::Warning(tpfx, "Source TTF font '%s' not found.", ttf_font.data());
         return false;
      }
      // Invoke through interpreter to avoid REve dependece on RGL.
      char command[8192];
      int cl = snprintf(command, 8192, "TGLSdfFontMaker::MakeFont(\"%s\", \"%s\");",
                        ttf_font.data(), base.data());
      if (cl < 0) {
         ::Warning(tpfx, "Error generating interpreter command for TGLSdfFontMaker::MakeFont(), ret=%d.", cl);
         return false;
      }
#ifdef WIN32
      while (--cl >= 0) if (command[cl] == '\\') command[cl] = '/';
#endif
      gROOT->ProcessLine(command);
      if (gSystem->AccessPathName(png.data()) || gSystem->AccessPathName(js.data())) {
         ::Warning(tpfx, "Creation of font '%s' failed.", font_name.data());
         return false;
      }
   }
   return true;
}
