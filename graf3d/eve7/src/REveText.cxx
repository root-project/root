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

////////////////////////////////////////////////////////////////////////////////
/// Make this text a button: a click that does not become a drag sends `mir`
/// to `target`, or to this element if null. The effect reaches every client.
/// A target that is an REveAunt is tracked, and its destruction clears the
/// action. Any other target is stored by id only.

void REveText::SetClickAction(const std::string &mir, REveElement *target)
{
   if (fClickAunt) {
      fClickAunt->RemoveNiece(this);
      fClickAunt = nullptr;
   }

   fClickMir = mir;
   fClickTargetId = target ? target->GetElementId() : 0;

   if (target) {
      if (auto au = dynamic_cast<REveAunt *>(target)) {
         au->AddNiece(this);
         fClickAunt = au;
      }
   }
   StampObjProps();
}

////////////////////////////////////////////////////////////////////////////////
/// Clear the click action when the click target is destroyed, so the button
/// does not keep an id that may be reused by an unrelated element.

void REveText::RemoveAunt(REveAunt *au)
{
   if (au == fClickAunt) {
      fClickAunt = nullptr;
      fClickMir.clear();
      fClickTargetId = 0;
      StampObjProps();
   }
   REveElement::RemoveAunt(au);
}

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
   j["fFontWeight"] = fFontWeight;
   j["fExtraBorder"] = fExtraBorder;
   j["fMode"] = fMode;
   j["fResizable"] = fResizable;
   j["fAlignH"] = fAlignH;
   j["fAlignV"] = fAlignV;
   j["fClickMir"] = fClickMir;
   j["fClickTargetId"] = fClickTargetId;
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
/// Text has no box: its extent is known only to the client, which lays it
/// out. It is not culled.

void REveText::ComputeBBox()
{
   BBoxNotInWorld();
}

#include "REveAssetDir.hxx"

#include "TSystem.h"
#include "TROOT.h"

#include <set>

using ROOT::Experimental::Internal::REveAssetDir;

std::string REveText::sSdfFontDir;

namespace {

////////////////////////////////////////////////////////////////////////////////
/// The SDF font directories, served under "sdf-fonts/". The fallback is the
/// directory of the fonts shipped with ROOT.

REveAssetDir &SdfFontDirs()
{
   static REveAssetDir s_dirs("sdf-fonts/");
   static bool s_have_fallback = false;
   if (!s_have_fallback) {
      s_have_fallback = true;
      s_dirs.AddFallbackDir(REveAssetDir::GetAssetsDir() + "sdf-fonts/");
   }
   return s_dirs;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Set location where SDF fonts and their metrics data are stored or are to be
/// created via the AssertSdfFont() static function.
/// The directory is created if it does not exist.
/// If `require_write_access` is true (default), write permission in the
/// directory is required, else read permission.
/// The directory replaces the ones added with AddSdfFontDir(). The fonts shipped
/// with ROOT are found after it.
/// Can be called before REveManager::Create().
/// Static function.

bool REveText::SetSdfFontDir(const std::string &dir, bool require_write_access)
{
   static const char* tpfx = "REveText::SetSdfFontDir";

   if (dir.empty()) {
      ::Error(tpfx, "Empty directory given.");
      return false;
   }

   std::string sanitized_dir = REveAssetDir::NormalizeDir(dir);
   if (gSystem->AccessPathName(sanitized_dir.data())) {
      if (gSystem->mkdir(sanitized_dir.data(), true)) {
         ::Error(tpfx, "Directory does not exist and mkdir failed for '%s", dir.data());
         return false;
      }
   }
   auto dir_perms = require_write_access ? kWritePermission : kReadPermission;
   if (gSystem->AccessPathName(sanitized_dir.data(), dir_perms) == false) {
      sSdfFontDir = sanitized_dir;
      auto &fonts = SdfFontDirs();
      fonts.SetWriteDir(sSdfFontDir);
      fonts.SetUserDir(sSdfFontDir);
      return true;
   } else {
      return false;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Add a directory fonts are read from, searched after the one of
/// SetSdfFontDir() and before the fonts shipped with ROOT. Fonts are not
/// generated into it. If no font directory is set, the default one is set first.
/// Can be called before REveManager::Create().
/// Static function.

bool REveText::AddSdfFontDir(const std::string &dir)
{
   static const char *tpfx = "REveText::AddSdfFontDir";

   std::string sanitized_dir = REveAssetDir::NormalizeDir(dir);
   if (sanitized_dir.empty() || gSystem->AccessPathName(sanitized_dir.data(), kReadPermission)) {
      ::Error(tpfx, "SDF font directory '%s' does not exist or is not readable.", dir.data());
      return false;
   }
   if (sSdfFontDir.empty())
      SetDefaultSdfFontDir();
   SdfFontDirs().AddUserDir(sanitized_dir);
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the default SDF font directory: sdf-fonts/ in the ui5 directory if it is
/// writable or can be created, else sdf-fonts/ in the current working directory.
/// The latter is created only when a font is generated into it.
/// The fonts shipped with ROOT, in eve7/assets/sdf-fonts/ of the ui5 directory,
/// are found after it.
/// Alternative fallback to /tmp or user's home directory is not attempted.

bool REveText::SetDefaultSdfFontDir()
{
   static const char *tpfx = "REveText::SetDefaultSdfFontDir";

   std::string dir = REveAssetDir::GetUi5Dir() + "/eve7/sdf-fonts/";
   bool usable = gSystem->AccessPathName(dir.c_str()) ? gSystem->mkdir(dir.c_str(), true) == 0
                                                      : !gSystem->AccessPathName(dir.c_str(), kWritePermission);
   if (usable) {
      ::Info(tpfx, "Using SDF font dir %s", dir.c_str());
   } else {
      dir = "./sdf-fonts/";
      ::Info(tpfx, "Using SDF font dir sdf-fonts/ in current directory");
   }

   sSdfFontDir = dir;
   auto &fonts = SdfFontDirs();
   fonts.SetWriteDir(dir);
   fonts.SetUserDir(dir);

   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if font exists, otherwise try to create it.
/// The font is looked up in the directories of SetSdfFontDir() and
/// AddSdfFontDir(), then in the fonts shipped with ROOT. If SDF font dir is not
/// yet set, it is set to the default, see SetDefaultSdfFontDir().
/// The font is present when the first directory that has its .png or its .js.gz
/// has both, as the server serves each file from the first directory that has it.
/// A missing font is generated into the directory of SetSdfFontDir().
/// In batch mode a missing font is not generated, as that needs a display.
/// Returns true if font files are present, false otherwise.
/// Static function.

bool REveText::AssertSdfFont(const std::string &font_name, const std::string &ttf_font)
{
   static const char* tpfx = "REveText::AssertSdfFont";

   if (sSdfFontDir.empty())
      SetDefaultSdfFontDir();

   auto &fonts = SdfFontDirs();
   std::string png = font_name + ".png";
   std::string js = font_name + ".js.gz";

   if (fonts.FindPair(png, js).empty()) {
      // A lone file, or two files in different directories, would be served as a mismatched pair.
      std::string at_png = fonts.Find(png), at_js = fonts.Find(js), split;
      auto where = [](const std::string &f, const std::string &p) {
         return f + (p.empty() ? " not found" : " found at '" + p + "'");
      };
      if (!at_png.empty() || !at_js.empty())
         split = " Its two files must be in one directory: " + where(png, at_png) + ", " + where(js, at_js) + ".";
      if (gSystem->AccessPathName(ttf_font.data())) {
         ::Warning(tpfx, "Source TTF font '%s' not found.%s", ttf_font.data(), split.data());
         return false;
      }
      // Generating a font needs a GL context, and TGLWidget throws without a display.
      if (gROOT->IsBatch()) {
         static std::set<std::string> s_warned_fonts;
         if (s_warned_fonts.insert(font_name).second) {
            if (split.empty())
               ::Warning(tpfx,
                         "SDF font '%s' is not in '%s' and is not generated in batch mode. "
                         "Generating it needs a GL context, which needs a display. "
                         "Run once with a display, or copy %s.png and %s.js.gz into the SDF font directory.",
                         font_name.data(), sSdfFontDir.data(), font_name.data(), font_name.data());
            else
               ::Warning(tpfx, "SDF font '%s' is not used and is not generated in batch mode.%s", font_name.data(),
                         split.data());
         }
         return false;
      }
      std::string base = fonts.WritePath(font_name);
      if (base.empty()) {
         ::Warning(tpfx, "Cannot create SDF font directory '%s', font '%s' not generated.", fonts.GetWriteDir().data(),
                   font_name.data());
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
      // The write directory is the first user directory, so the generated pair is found and served first.
      if (fonts.FindPair(png, js).empty()) {
         ::Warning(tpfx, "Creation of font '%s' failed.", font_name.data());
         return false;
      }
   }
   return true;
}
