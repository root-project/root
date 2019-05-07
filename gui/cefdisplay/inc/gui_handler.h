/// \file gui_handler.h
/// \ingroup WebGui
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_cef_gui_handler
#define ROOT_cef_gui_handler

#include "base_handler.h"

class GuiHandler : public BaseHandler {
public:
   explicit GuiHandler(THttpServer *serv = nullptr, bool use_views = false);

   // CefDisplayHandler methods:
   virtual void OnTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title) OVERRIDE;

private:
   // Platform-specific initialization
   void PlatformInit();

   // Platform-specific implementation.
   void PlatformTitleChange(CefRefPtr<CefBrowser> browser, const CefString &title);

   // True if the application is using the Views framework.
   const bool use_views_;

   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(GuiHandler);
   DISALLOW_COPY_AND_ASSIGN(GuiHandler);
};

#endif // ROOT_cef_simple_handler
