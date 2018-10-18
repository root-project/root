/// \file ROOT/RWebDisplayHandle.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2018-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RWebDisplayHandle
#define ROOT7_RWebDisplayHandle

#include <string>
#include <map>
#include <memory>
#include <functional>

#include "TString.h"

class THttpServer;

namespace ROOT {
namespace Experimental {

class RWebDisplayHandle {

protected:
   class Creator {
   public:
      virtual std::unique_ptr<RWebDisplayHandle>
      ShowURL(THttpServer *serv, const std::string &url, bool batch, int width, int height) = 0;
      virtual ~Creator() = default;
   };

   class BrowserCreator : public Creator {
   protected:
      std::string fProg;  ///< browser executable
      std::string fExec;  ///< standard execute line
      std::string fBatchExec; ///< batch execute line

      void TestProg(const std::string &nexttry, bool check_std_paths = false);

      virtual std::string MakeProfile(TString &, bool) { return ""; }

   public:

      BrowserCreator(bool dflt = true, const std::string &where_arg = "");

      std::unique_ptr<RWebDisplayHandle>
      ShowURL(THttpServer *serv, const std::string &url, bool batch, int width, int height) override;

      virtual ~BrowserCreator() = default;
   };

   class ChromeCreator : public BrowserCreator {
   public:
      ChromeCreator();
      virtual ~ChromeCreator() = default;
   };

   class FirefoxCreator : public BrowserCreator {
   public:
      FirefoxCreator();
      virtual ~FirefoxCreator() = default;

      std::string MakeProfile(TString &exec, bool batch) override;
   };

   std::string fUrl; ///!< URL used to launch display

   static std::map<std::string, std::unique_ptr<Creator>> &GetMap();

   static std::unique_ptr<Creator> &FindCreator(const std::string &name, const std::string &libname = "");

   static void TestProg(TString &prog, const std::string &nexttry);

public:

   /// Function should return URL for the widget dislpayed - local or http
   using CreateUrlFunc_t = std::function<std::string(bool)>;

   RWebDisplayHandle(const std::string &url) : fUrl(url) {}

   std::string GetUrl() const { return fUrl; }

   static std::unique_ptr<RWebDisplayHandle> Display(const std::string &where, CreateUrlFunc_t func, THttpServer *serv = nullptr, bool batch_mode = false, int width = 0, int height = 0);

   virtual ~RWebDisplayHandle() = default;
};

}
}



#endif
