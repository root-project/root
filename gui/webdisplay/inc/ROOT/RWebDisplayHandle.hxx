// Author: Sergey Linev <s.linev@gsi.de>
// Date: 2018-10-17
// Warning: This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebDisplayHandle
#define ROOT7_RWebDisplayHandle

#include <ROOT/RWebDisplayArgs.hxx>

#include <string>
#include <map>
#include <memory>
#include <vector>
#include "TString.h"

namespace ROOT {

class RWebDisplayHandle {

   std::string fUrl; ///!< URL used to launch display

   std::string fContent; ///!< page content

protected:

   class Creator {
   public:
      virtual std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) = 0;
      virtual bool IsActive() const { return true; }
      virtual ~Creator() = default;
      virtual bool IsSnapBrowser() const { return false; }
   };

   class BrowserCreator : public Creator {
   protected:
      std::string fProg;  ///< browser executable
      std::string fExec;  ///< standard execute line
      std::string fHeadlessExec; ///< headless execute line
      std::string fBatchExec; ///< batch execute line
      void TestProg(const std::string &nexttry, bool check_std_paths = false);
      virtual void ProcessGeometry(std::string &, const RWebDisplayArgs &) {}
      virtual std::string MakeProfile(std::string &, bool) { return ""; }
   public:
      BrowserCreator(bool custom = true, const std::string &exec = "");
      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override;
      ~BrowserCreator() override = default;
      static FILE *TemporaryFile(TString &name, int use_home_dir = 0, const char *suffix = nullptr);
   };

   class SafariCreator : public BrowserCreator {
   public:
      SafariCreator();
      ~SafariCreator() override = default;
      bool IsActive() const override;
   };

   class ChromeCreator : public BrowserCreator {
      bool fEdge{false};
      std::string fEnvPrefix; // rc parameters prefix
      int fChromeVersion{-1}; // major version in chrome browser
   public:
      ChromeCreator(bool is_edge = false);
      ~ChromeCreator() override = default;
      bool IsActive() const override { return !fProg.empty(); }
      bool IsSnapBrowser() const override { return fProg == "/snap/bin/chromium"; }
      void ProcessGeometry(std::string &, const RWebDisplayArgs &) override;
      std::string MakeProfile(std::string &exec, bool) override;
   };

   class FirefoxCreator : public BrowserCreator {
   public:
      FirefoxCreator();
      ~FirefoxCreator() override = default;
      bool IsActive() const override { return !fProg.empty(); }
      bool IsSnapBrowser() const override { return fProg == "/snap/bin/firefox"; }
      void ProcessGeometry(std::string &, const RWebDisplayArgs &) override;
      std::string MakeProfile(std::string &exec, bool batch) override;
   };

   static std::map<std::string, std::unique_ptr<Creator>> &GetMap();

   static std::unique_ptr<Creator> &FindCreator(const std::string &name, const std::string &libname = "");

   static bool CheckIfCanProduceImages(RWebDisplayArgs &args);

public:

   /// constructor
   RWebDisplayHandle(const std::string &url) : fUrl(url) {}

   /// required virtual destructor for correct cleanup at the end
   virtual ~RWebDisplayHandle() = default;

   /// returns url of start web display
   const std::string &GetUrl() const { return fUrl; }

   /// set content
   void SetContent(const std::string &cont) { fContent = cont; }
   /// get content
   const std::string &GetContent() const { return fContent; }

   /// resize web window - if possible
   virtual bool Resize(int, int) { return false; }

   /// remove file which was used to startup widget - if possible
   virtual void RemoveStartupFiles() {}

   static bool NeedHttpServer(const RWebDisplayArgs &args);

   static std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args);

   static bool DisplayUrl(const std::string &url);

   static bool CanProduceImages(const std::string &browser = "");

   static std::string GetImageFormat(const std::string &fname);

   static bool ProduceImage(const std::string &fname, const std::string &json, int width = 800, int height = 600, const char *batch_file = nullptr);

   static bool ProduceImages(const std::string &fname, const std::vector<std::string> &jsons, const std::vector<int> &widths, const std::vector<int> &heights, const char *batch_file = nullptr);

   static std::vector<std::string> ProduceImagesNames(const std::string &fname, unsigned nfiles = 1);

   static bool ProduceImages(const std::vector<std::string> &fnames, const std::vector<std::string> &jsons, const std::vector<int> &widths, const std::vector<int> &heights, const char *batch_file = nullptr);

};

} // namespace ROOT

#endif
