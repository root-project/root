/// \file ROOT/RCefWebDisplayHandle.hxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2020-08-21
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RCefWebDisplayHandle
#define ROOT7_RCefWebDisplayHandle

#include <ROOT/RWebDisplayHandle.hxx>

#include "simple_app.h"

namespace ROOT {
namespace Experimental {

class RCefWebDisplayHandle : public RWebDisplayHandle {
protected:
   class CefCreator : public Creator {

      CefRefPtr<SimpleApp> fCefApp;
   public:
      CefCreator() = default;
      virtual ~CefCreator() = default;

      std::unique_ptr<RWebDisplayHandle> Display(const RWebDisplayArgs &args) override;
   };

   enum EValidValues { kValid = 0x3C3C3C3C, kInvalid = 0x92929292 };

   unsigned fValid{kValid};  ///< used to verify if instance valid or not

   CefRefPtr<CefBrowser> fBrowser; ///< associated browser

public:
   RCefWebDisplayHandle(const std::string &url) : RWebDisplayHandle(url) {}

   virtual ~RCefWebDisplayHandle();

   bool IsValid() const { return fValid == kValid; }

   static void AddCreator();
};

} // namespace Experimental
} // namepsace ROOT


#endif
