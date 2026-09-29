// @(#)root/eve7:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveLogo
#define ROOT7_REveLogo

#include <ROOT/REveElement.hxx>

#include <string>

namespace ROOT {
namespace Experimental {


class REveLogo : public REveElement
{
public:
   /// Where the image comes from: a directory registered with SetImageDir()
   /// (the default), ui5/eve7/textures/, or an absolute URL fetched by the
   /// browser. The constructor selects kRemote for a URL; SetFile() does not.
   enum EImageSource_e { kImageDir = 0, kTextures, kRemote };

private:
   REveLogo(const REveLogo &) = delete;
   REveLogo &operator=(const REveLogo &) = delete;

protected:
   std::string fFile;         ///< file name or URL, interpreted according to fSource

   Float_t fPosX{0.06};       ///< position in the (0,1) overlay box; the image is centred on it
   Float_t fPosY{0.92};
   Float_t fSize{96};         ///< height in CSS pixels; width follows the image aspect
   Float_t fOpacity{0.8};     ///< resting opacity; the client brightens it on hover

   Bool_t fResizable{true};
   EImageSource_e fSource{kImageDir};

   static std::string sImageDir;

public:
   REveLogo(std::string_view file, const Text_t *n = "REveLogo", const Text_t *t = "");
   ~REveLogo() override {}

   static bool IsRemote(const std::string &f);

   static bool SetImageDir(std::string_view dir);
   static const std::string &GetImageDir() { return sImageDir; }

   /// Either a file name inside the registered image directory, or an absolute
   /// URL the client fetches directly. A remote URL fails silently on CORS,
   /// mixed content or a bad certificate; serve locally via SetImageDir().
   const std::string &GetFile() const { return fFile; }
   void SetFile(std::string_view f) { fFile = f; StampObjProps(); }

   Float_t GetPosX() const { return fPosX; }
   Float_t GetPosY() const { return fPosY; }
   void SetPosition(Float_t x, Float_t y) { fPosX = x; fPosY = y; StampObjProps(); }

   Float_t GetSize() const { return fSize; }
   void SetSize(Float_t s) { fSize = s; StampObjProps(); }

   Float_t GetOpacity() const { return fOpacity; }
   void SetOpacity(Float_t o) { fOpacity = o; StampObjProps(); }

   EImageSource_e GetSource() const { return fSource; }
   void SetSource(EImageSource_e s) { fSource = s; StampObjProps(); }

   Bool_t GetResizable() const { return fResizable; }
   void SetResizable(Bool_t r) { fResizable = r; StampObjProps(); }

   Int_t WriteCoreJson(nlohmann::json &j, Int_t rnr_offset) override;
   void BuildRenderData() override;
};

} // namespace Experimental
} // namespace ROOT

#endif
