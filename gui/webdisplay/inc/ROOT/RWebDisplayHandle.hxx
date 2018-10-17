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

#include "TString.h"

class THttpServer;

namespace ROOT {
namespace Experimental {

class RWebWindow;
class RWebWindowsManager;

class RWebDisplayHandle {
   friend class RWebWindowsManager;

protected:
   class Creator {
   public:
      virtual std::unique_ptr<RWebDisplayHandle>
      Make(THttpServer *serv, const std::string &url, bool batch, int width, int height) = 0;
      virtual ~Creator() = default;
   };

   std::string fUrl; ///!< URL used to launch display

   static std::map<std::string, std::unique_ptr<Creator>> &GetMap();

   static std::unique_ptr<Creator> &FindCreator(const std::string &name, const std::string &libname = "");

   static void TestProg(TString &prog, const std::string &nexttry);

   static unsigned DisplayWindow(RWebWindow &win, bool batch_mode, const std::string &where);

public:

   RWebDisplayHandle(const std::string &url) : fUrl(url) {}

   std::string GetUrl() const { return fUrl; }

   virtual ~RWebDisplayHandle() = default;
};

}
}



#endif
