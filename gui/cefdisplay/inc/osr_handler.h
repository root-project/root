/// \file osr_handler.h
/// \ingroup WebUI
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

#ifndef ROOT_cef_osr_handler
#define ROOT_cef_osr_handler

#include "base_handler.h"

#include "include/cef_version.h"

/// Class used to handle off-screen application and should emulate some render requests

class OsrHandler : public BaseHandler, public CefRenderHandler {
public:
   explicit OsrHandler(THttpServer *serv = nullptr);

   // CefClient methods:
   virtual CefRefPtr<CefRenderHandler> GetRenderHandler() OVERRIDE { return this; }

   // CefRenderHandler methods.
   virtual bool GetRootScreenRect(CefRefPtr<CefBrowser> browser, CefRect &rect) OVERRIDE;
#if CEF_COMMIT_NUMBER > 1894
   virtual void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) OVERRIDE;
#else
   virtual bool GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) OVERRIDE;
#endif
   virtual bool GetScreenPoint(CefRefPtr<CefBrowser> browser, int viewX, int viewY, int &screenX,
                               int &screenY) OVERRIDE;
   virtual bool GetScreenInfo(CefRefPtr<CefBrowser> browser, CefScreenInfo &screen_info) OVERRIDE;
   virtual void OnPopupShow(CefRefPtr<CefBrowser> browser, bool show) OVERRIDE;
   virtual void OnPopupSize(CefRefPtr<CefBrowser> browser, const CefRect &rect) OVERRIDE;
   virtual void OnPaint(CefRefPtr<CefBrowser> browser, CefRenderHandler::PaintElementType type,
                        const CefRenderHandler::RectList &dirtyRects, const void *buffer, int width,
                        int height) OVERRIDE;
   virtual void OnCursorChange(CefRefPtr<CefBrowser> browser, CefCursorHandle cursor, CursorType type,
                               const CefCursorInfo &custom_cursor_info) OVERRIDE;
   virtual bool StartDragging(CefRefPtr<CefBrowser> browser, CefRefPtr<CefDragData> drag_data,
                              CefRenderHandler::DragOperationsMask allowed_ops, int x, int y) OVERRIDE;
   virtual void UpdateDragCursor(CefRefPtr<CefBrowser> browser, CefRenderHandler::DragOperation operation) OVERRIDE;
   virtual void OnImeCompositionRangeChanged(CefRefPtr<CefBrowser> browser, const CefRange &selection_range,
                                             const CefRenderHandler::RectList &character_bounds) OVERRIDE;

private:
   // Include the default reference counting implementation.
   IMPLEMENT_REFCOUNTING(OsrHandler);
   DISALLOW_COPY_AND_ASSIGN(OsrHandler);
};

#endif // ROOT_cef_osr_handler
