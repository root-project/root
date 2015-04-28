// @(#)root/x11:$Id$
// Author: Fons Rademakers   28/12/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGX11 (GUI related part)                                             //
//                                                                      //
// This class is the basic interface to the X11 graphics system. It is  //
// an implementation of the abstract TVirtualX class. The companion     //
// class for Win32 is TGWin32.                                          //
//                                                                      //
// This file contains the implementation of the GUI methods of the      //
// TGX11 class. Most of the methods are used by the machine independent //
// GUI classes (libGUI.so).                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <limits.h>
#include <unistd.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xatom.h>
#include <X11/cursorfont.h>
#include <X11/keysym.h>
#include <X11/xpm.h>

#include "TGX11.h"
#include "TROOT.h"
#include "TError.h"
#include "TSystem.h"
#include "TException.h"
#include "TClassTable.h"
#include "KeySymbols.h"
#include "TEnv.h"

#include <X11/extensions/shape.h>

//---- MWM Hints stuff

struct MWMHintsProperty_t {
   Handle_t  fFlags;
   Handle_t  fFunctions;
   Handle_t  fDecorations;
   Int_t     fInputMode;
};

//---- hints

const ULong_t kMWMHintsFunctions   = BIT(0);
const ULong_t kMWMHintsDecorations = BIT(1);
const ULong_t kMWMHintsInputMode   = BIT(2);

const Int_t kPropMotifWMHintsElements = 4;
const Int_t kPropMWMHintElements      = kPropMotifWMHintsElements;


//---- Key symbol mapping

struct KeySymbolMap_t {
   KeySym    fXKeySym;
   EKeySym   fKeySym;
};

//---- Mapping table of all non-trivial mappings (the ASCII keys map
//---- one to one so are not included)

static KeySymbolMap_t gKeyMap[] = {
   { XK_Escape,          kKey_Escape },
   { XK_Tab,             kKey_Tab },
#ifndef XK_ISO_Left_Tab
   { 0xFE20,             kKey_Backtab },
#else
   { XK_ISO_Left_Tab,    kKey_Backtab },
#endif
   { XK_BackSpace,       kKey_Backspace },
   { XK_Return,          kKey_Return },
   { XK_Insert,          kKey_Insert },
   { XK_Delete,          kKey_Delete },
   { XK_Clear,           kKey_Delete },
   { XK_Pause,           kKey_Pause },
   { XK_Print,           kKey_Print },
   { 0x1005FF60,         kKey_SysReq },             // hardcoded Sun SysReq
   { 0x1007ff00,         kKey_SysReq },             // hardcoded X386 SysReq
   { XK_Home,            kKey_Home },               // cursor movement
   { XK_End,             kKey_End },
   { XK_Left,            kKey_Left },
   { XK_Up,              kKey_Up },
   { XK_Right,           kKey_Right },
   { XK_Down,            kKey_Down },
   { XK_Prior,           kKey_Prior },
   { XK_Next,            kKey_Next },
   { XK_Shift_L,         kKey_Shift },              // modifiers
   { XK_Shift_R,         kKey_Shift },
   { XK_Shift_Lock,      kKey_Shift },
   { XK_Control_L,       kKey_Control },
   { XK_Control_R,       kKey_Control },
   { XK_Meta_L,          kKey_Meta },
   { XK_Meta_R,          kKey_Meta },
   { XK_Alt_L,           kKey_Alt },
   { XK_Alt_R,           kKey_Alt },
   { XK_Caps_Lock,       kKey_CapsLock },
   { XK_Num_Lock,        kKey_NumLock },
   { XK_Scroll_Lock,     kKey_ScrollLock },
   { XK_KP_Space,        kKey_Space },              // numeric keypad
   { XK_KP_Tab,          kKey_Tab },
   { XK_KP_Enter,        kKey_Enter },
   { XK_KP_F1,           kKey_F1 },
   { XK_KP_F2,           kKey_F2 },
   { XK_KP_F3,           kKey_F3 },
   { XK_KP_F4,           kKey_F4 },
   { XK_KP_Home,         kKey_Home },
   { XK_KP_Left,         kKey_Left },
   { XK_KP_Up,           kKey_Up },
   { XK_KP_Right,        kKey_Right },
   { XK_KP_Down,         kKey_Down },
   { XK_KP_Prior,        kKey_Prior },
   { XK_KP_Page_Up,      kKey_PageUp },
   { XK_KP_Next,         kKey_Next },
   { XK_KP_Page_Down,    kKey_PageDown },
   { XK_KP_End,          kKey_End },
   { XK_KP_Begin,        kKey_Home },
   { XK_KP_Insert,       kKey_Insert },
   { XK_KP_Delete,       kKey_Delete },
   { XK_KP_Equal,        kKey_Equal },
   { XK_KP_Multiply,     kKey_Asterisk },
   { XK_KP_Add,          kKey_Plus },
   { XK_KP_Separator,    kKey_Comma },
   { XK_KP_Subtract,     kKey_Minus },
   { XK_KP_Decimal,      kKey_Period },
   { XK_KP_Divide,       kKey_Slash },
   { 0,                  (EKeySym) 0 }
};

struct RXGCValues:XGCValues{};
struct RXColor:XColor{};
struct RXpmAttributes:XpmAttributes{};
struct RXSetWindowAttributes:XSetWindowAttributes{};
struct RVisual:Visual{};

//______________________________________________________________________________
inline void SplitLong(Long_t ll, Long_t &i1, Long_t &i2)
{
   union { Long_t l; Int_t i[2]; } conv;

   conv.l = ll;
   i1 = conv.i[0];
   i2 = conv.i[1];
}

//______________________________________________________________________________
inline void AsmLong(Long_t i1, Long_t i2, Long_t &ll)
{
   union { Long_t l; Int_t i[2]; } conv;

   conv.i[0] = (Int_t) i1;
   conv.i[1] = (Int_t) i2;
   ll = conv.l;
}

//______________________________________________________________________________
static Int_t RootX11ErrorHandler(Display *disp, XErrorEvent *err)
{
   // Handle X11 error.

   char msg[80];
   XGetErrorText(disp, err->error_code, msg, 80);

   // force segV. to allow backtracing the error with gdb
   if (gDebug == (Long_t)gVirtualX) {
      gSystem->ProcessEvents();
      ::Error("RootX11ErrorHandler", "%s (XID: %u, XREQ: %u)", msg,
               (UInt_t)err->resourceid, err->request_code);
      int *kil = (int*)1;
      delete kil;
      return 0;
   }

   if (!err->resourceid) return 0;

   TObject *w = (TObject *)gROOT->ProcessLineFast(Form("gClient->GetWindowById(%lu)", (ULong_t)err->resourceid));

   if (!w) {
      ::Error("RootX11ErrorHandler", "%s (XID: %u, XREQ: %u)", msg,
               (UInt_t)err->resourceid, err->request_code);
   } else {
      ::Error("RootX11ErrorHandler", "%s (%s XID: %u, XREQ: %u)", msg, w->ClassName(),
               (UInt_t)err->resourceid, err->request_code);
      w->Print("tree");
   }
   if (TROOT::Initialized()) {
      //Getlinem(kInit, "Root > ");
      Throw(2);
   }
   return 0;
}

//______________________________________________________________________________
static Int_t RootX11IOErrorHandler(Display *)
{
   // Handle X11 I/O error (happens when connection to display server
   // is broken).

   ::Error("RootX11IOErrorHandler", "fatal X11 error (connection to server lost?!)");
   fprintf(stderr,"\n**** Save data and exit application ****\n\n");
   // delete X connection handler (to avoid looping in TSystem::DispatchOneEvent())
   if (gXDisplay && gSystem) {
      gSystem->RemoveFileHandler(gXDisplay);
      SafeDelete(gXDisplay);
   }
   if (TROOT::Initialized()) {
      //Getlinem(kInit, "Root > ");
      Throw(2);
   }
   return 0;
}


//______________________________________________________________________________
void TGX11::MapWindow(Window_t id)
{
   // Map window on screen.

   if (!id) return;

   XMapWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MapSubwindows(Window_t id)
{
   // Map sub windows.

   if (!id) return;

   XMapSubwindows((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.

   if (!id) return;

   XMapRaised((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::UnmapWindow(Window_t id)
{
   // Unmap window from screen.

   if (!id) return;

   XUnmapWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::DestroyWindow(Window_t id)
{
   // Destroy window.

   if (!id) return;

   XDestroyWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::DestroySubwindows(Window_t id)
{
   // Destroy subwindows of this window.

   if (!id) return;

   XDestroySubwindows((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.

   if (!id) return;

   XRaiseWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.

   if (!id) return;

   XLowerWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.

   if (!id) return;

   XMoveWindow((Display*)fDisplay, (Window) id, x, y);
}

//______________________________________________________________________________
void TGX11::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize a window.

   if (!id) return;

   XMoveResizeWindow((Display*)fDisplay, (Window) id, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.

   if (!id) return;

   // protect against potential negative values
   if (w >= (UInt_t)INT_MAX || h >= (UInt_t)INT_MAX)
      return;
   XResizeWindow((Display*)fDisplay, (Window) id, w, h);
}

//______________________________________________________________________________
void TGX11::IconifyWindow(Window_t id)
{
   // Iconify the window.

   if (!id) return;

   XIconifyWindow((Display*)fDisplay, (Window) id, fScreenNumber);
}

//______________________________________________________________________________
void TGX11::ReparentWindow(Window_t id, Window_t pid, Int_t x, Int_t y)
{
   // Reparent window to new parent window at position (x,y).

   if (!id) return;

   XReparentWindow((Display*)fDisplay, (Window) id, (Window) pid, x, y);
}

//______________________________________________________________________________
void TGX11::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.

   if (!id) return;

   XSetWindowBackground((Display*)fDisplay, (Window) id, color);
}

//______________________________________________________________________________
void TGX11::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.

   if (!id) return;

   XSetWindowBackgroundPixmap((Display*)fDisplay, (Window) id, (Pixmap) pxm);
}

//______________________________________________________________________________
Window_t TGX11::CreateWindow(Window_t parent, Int_t x, Int_t y,
                             UInt_t w, UInt_t h, UInt_t border,
                             Int_t depth, UInt_t clss,
                             void *visual, SetWindowAttributes_t *attr, UInt_t)
{
   // Return handle to newly created X window.

   RXSetWindowAttributes xattr;
   ULong_t              xmask = 0;

   if (attr)
      MapSetWindowAttributes(attr, xmask, xattr);

   if (depth == 0)
      depth = fDepth;
   if (visual == 0)
      visual = fVisual;
   if (fColormap && !(xmask & CWColormap)) {
      xmask |= CWColormap;
      xattr.colormap = fColormap;
   }
   if ((Window)parent == fRootWin && fRootWin != fVisRootWin) {
      xmask |= CWBorderPixel;
      xattr.border_pixel = fBlackPixel;
   }

   return (Window_t) XCreateWindow((Display*)fDisplay, (Window) parent, x, y,
                                   w, h, border, depth, clss, (Visual*)visual,
                                   xmask, &xattr);
}

//______________________________________________________________________________
void TGX11::MapEventMask(UInt_t &emask, UInt_t &xemask, Bool_t tox)
{
   // Map event mask to or from X.

   if (tox) {
      Long_t lxemask = NoEventMask;
      if ((emask & kKeyPressMask))
         lxemask |= KeyPressMask;
      if ((emask & kKeyReleaseMask))
         lxemask |= KeyReleaseMask;
      if ((emask & kButtonPressMask))
         lxemask |= ButtonPressMask;
      if ((emask & kButtonReleaseMask))
         lxemask |= ButtonReleaseMask;
      if ((emask & kPointerMotionMask))
         lxemask |= PointerMotionMask;
      if ((emask & kButtonMotionMask))
         lxemask |= ButtonMotionMask;
      if ((emask & kExposureMask))
         lxemask |= ExposureMask;
      if ((emask & kStructureNotifyMask))
         lxemask |= StructureNotifyMask;
      if ((emask & kEnterWindowMask))
         lxemask |= EnterWindowMask;
      if ((emask & kLeaveWindowMask))
         lxemask |= LeaveWindowMask;
      if ((emask & kFocusChangeMask))
         lxemask |= FocusChangeMask;
      if ((emask & kOwnerGrabButtonMask))
         lxemask |= OwnerGrabButtonMask;
      if ((emask & kColormapChangeMask))
         lxemask |= ColormapChangeMask;
      xemask = (UInt_t)lxemask;
   } else {
      emask = 0;
      if ((xemask & KeyPressMask))
         emask |= kKeyPressMask;
      if ((xemask & KeyReleaseMask))
         emask |= kKeyReleaseMask;
      if ((xemask & ButtonPressMask))
         emask |= kButtonPressMask;
      if ((xemask & ButtonReleaseMask))
         emask |= kButtonReleaseMask;
      if ((xemask & PointerMotionMask))
         emask |= kPointerMotionMask;
      if ((xemask & ButtonMotionMask))
         emask |= kButtonMotionMask;
      if ((xemask & ExposureMask))
         emask |= kExposureMask;
      if ((xemask & StructureNotifyMask))
         emask |= kStructureNotifyMask;
      if ((xemask & EnterWindowMask))
         emask |= kEnterWindowMask;
      if ((xemask & LeaveWindowMask))
         emask |= kLeaveWindowMask;
      if ((xemask & FocusChangeMask))
         emask |= kFocusChangeMask;
      if ((xemask & OwnerGrabButtonMask))
         emask |= kOwnerGrabButtonMask;
      if ((xemask & ColormapChangeMask))
         emask |= kColormapChangeMask;
   }
}

//______________________________________________________________________________
void TGX11::MapSetWindowAttributes(SetWindowAttributes_t *attr,
                                   ULong_t &xmask, RXSetWindowAttributes &xattr)
{
   // Map a SetWindowAttributes_t to a XSetWindowAttributes structure.

   Mask_t mask = attr->fMask;
   xmask = 0;

   if ((mask & kWABackPixmap)) {
      xmask |= CWBackPixmap;
      if (attr->fBackgroundPixmap == kNone)
         xattr.background_pixmap = None;
      else if (attr->fBackgroundPixmap == kParentRelative)
         xattr.background_pixmap = ParentRelative;
      else
         xattr.background_pixmap = (Pixmap)attr->fBackgroundPixmap;
   }
   if ((mask & kWABackPixel)) {
      xmask |= CWBackPixel;
      xattr.background_pixel = attr->fBackgroundPixel;
   }
   if ((mask & kWABorderPixmap)) {
      xmask |= CWBorderPixmap;
      xattr.border_pixmap = (Pixmap)attr->fBorderPixmap;
   }
   if ((mask & kWABorderPixel)) {
      xmask |= CWBorderPixel;
      xattr.border_pixel = attr->fBorderPixel;
   }
   if ((mask & kWABitGravity)) {
      xmask |= CWBitGravity;
      xattr.bit_gravity = attr->fBitGravity;  //assume ident mapping (rdm)
   }
   if ((mask & kWAWinGravity)) {
      xmask |= CWWinGravity;
      xattr.win_gravity = attr->fWinGravity;  // assume ident mapping (rdm)
   }
   if ((mask & kWABackingStore)) {
      xmask |= CWBackingStore;
      if (attr->fBackingStore == kNotUseful)
         xattr.backing_store = NotUseful;
      else if (attr->fBackingStore == kWhenMapped)
         xattr.backing_store = WhenMapped;
      else if (attr->fBackingStore == kAlways)
         xattr.backing_store = Always;
      else
         xattr.backing_store = attr->fBackingStore;
   }
   if ((mask & kWABackingPlanes)) {
      xmask |= CWBackingPlanes;
      xattr.backing_planes = attr->fBackingPlanes;
   }
   if ((mask & kWABackingPixel)) {
      xmask |= CWBackingPixel;
      xattr.backing_pixel = attr->fBackingPixel;
   }
   if ((mask & kWAOverrideRedirect)) {
      xmask |= CWOverrideRedirect;
      xattr.override_redirect = attr->fOverrideRedirect;
   }
   if ((mask & kWASaveUnder)) {
      xmask |= CWSaveUnder;
      xattr.save_under = (Bool)attr->fSaveUnder;
   }
   if ((mask & kWAEventMask)) {
      xmask |= CWEventMask;
      UInt_t xmsk, msk = (UInt_t) attr->fEventMask;
      MapEventMask(msk, xmsk, kTRUE);
      xattr.event_mask = xmsk;
   }
   if ((mask & kWADontPropagate)) {
      xmask |= CWDontPropagate;
      xattr.do_not_propagate_mask = attr->fDoNotPropagateMask;
   }
   if ((mask & kWAColormap)) {
      xmask |= CWColormap;
      xattr.colormap = (Colormap)attr->fColormap;
   }
   if ((mask & kWACursor)) {
      xmask |= CWCursor;
      if (attr->fCursor == kNone)
         xattr.cursor = None;
      else
         xattr.cursor = (Cursor)attr->fCursor;
   }
}

//______________________________________________________________________________
void TGX11::MapGCValues(GCValues_t &gval,
                        ULong_t &xmask, RXGCValues &xgval, Bool_t tox)
{
   // Map a GCValues_t to a XCGValues structure if tox is true. Map
   // the other way in case tox is false.

   if (tox) {
      // map GCValues_t to XGCValues
      Mask_t mask = gval.fMask;
      xmask = 0;

      if ((mask & kGCFunction)) {
         xmask |= GCFunction;
         xgval.function = gval.fFunction;   // ident mapping
      }
      if ((mask & kGCPlaneMask)) {
         xmask |= GCPlaneMask;
         xgval.plane_mask = gval.fPlaneMask;
      }
      if ((mask & kGCForeground)) {
         xmask |= GCForeground;
         xgval.foreground = gval.fForeground;
      }
      if ((mask & kGCBackground)) {
         xmask |= GCBackground;
         xgval.background = gval.fBackground;
      }
      if ((mask & kGCLineWidth)) {
         xmask |= GCLineWidth;
         xgval.line_width = gval.fLineWidth;
      }
      if ((mask & kGCLineStyle)) {
         xmask |= GCLineStyle;
         xgval.line_style = gval.fLineStyle;   // ident mapping
      }
      if ((mask & kGCCapStyle)) {
         xmask |= GCCapStyle;
         xgval.cap_style = gval.fCapStyle;     // ident mapping
      }
      if ((mask & kGCJoinStyle)) {
         xmask |= GCJoinStyle;
         xgval.join_style = gval.fJoinStyle;   // ident mapping
      }
      if ((mask & kGCFillStyle)) {
         xmask |= GCFillStyle;
         xgval.fill_style = gval.fFillStyle;   // ident mapping
      }
      if ((mask & kGCFillRule)) {
         xmask |= GCFillRule;
         xgval.fill_rule = gval.fFillRule;     // ident mapping
      }
      if ((mask & kGCTile)) {
         xmask |= GCTile;
         xgval.tile = (Pixmap) gval.fTile;
      }
      if ((mask & kGCStipple)) {
         xmask |= GCStipple;
         xgval.stipple = (Pixmap) gval.fStipple;
      }
      if ((mask & kGCTileStipXOrigin)) {
         xmask |= GCTileStipXOrigin;
         xgval.ts_x_origin = gval.fTsXOrigin;
      }
      if ((mask & kGCTileStipYOrigin)) {
         xmask |= GCTileStipYOrigin;
         xgval.ts_y_origin = gval.fTsYOrigin;
      }
      if ((mask & kGCFont)) {
         if (!fHasXft) {
            xmask |= GCFont;
            xgval.font = (Font) gval.fFont;
         }
      }
      if ((mask & kGCSubwindowMode)) {
         xmask |= GCSubwindowMode;
         xgval.subwindow_mode = gval.fSubwindowMode;  // ident mapping
      }
      if ((mask & kGCGraphicsExposures)) {
         xmask |= GCGraphicsExposures;
         xgval.graphics_exposures = (Bool) gval.fGraphicsExposures;
      }
      if ((mask & kGCClipXOrigin)) {
         xmask |= GCClipXOrigin;
         xgval.clip_x_origin = gval.fClipXOrigin;
      }
      if ((mask & kGCClipYOrigin)) {
         xmask |= GCClipYOrigin;
         xgval.clip_y_origin = gval.fClipYOrigin;
      }
      if ((mask & kGCClipMask)) {
         xmask |= GCClipMask;
         xgval.clip_mask = (Pixmap) gval.fClipMask;
      }
      if ((mask & kGCDashOffset)) {
         xmask |= GCDashOffset;
         xgval.dash_offset = gval.fDashOffset;
      }
      if ((mask & kGCDashList)) {
         xmask |= GCDashList;
         xgval.dashes = gval.fDashes[0];
      }
      if ((mask & kGCArcMode)) {
         xmask |= GCArcMode;
         xgval.arc_mode = gval.fArcMode;   // ident mapping
      }

   } else {
      // map XValues to GCValues_t
      Mask_t mask = 0;

      if ((xmask & GCFunction)) {
         mask |= kGCFunction;
         gval.fFunction = (EGraphicsFunction) xgval.function; // ident mapping
      }
      if ((xmask & GCPlaneMask)) {
         mask |= kGCPlaneMask;
         gval.fPlaneMask = xgval.plane_mask;
      }
      if ((xmask & GCForeground)) {
         mask |= kGCForeground;
         gval.fForeground = xgval.foreground;
      }
      if ((xmask & GCBackground)) {
         mask |= kGCBackground;
         gval.fBackground = xgval.background;
      }
      if ((xmask & GCLineWidth)) {
         mask |= kGCLineWidth;
         gval.fLineWidth = xgval.line_width;
      }
      if ((xmask & GCLineStyle)) {
         mask |= kGCLineStyle;
         gval.fLineStyle = xgval.line_style;   // ident mapping
      }
      if ((xmask & GCCapStyle)) {
         mask |= kGCCapStyle;
         gval.fCapStyle = xgval.cap_style;     // ident mapping
      }
      if ((xmask & GCJoinStyle)) {
         mask |= kGCJoinStyle;
         gval.fJoinStyle = xgval.join_style;   // ident mapping
      }
      if ((xmask & GCFillStyle)) {
         mask |= kGCFillStyle;
         gval.fFillStyle = xgval.fill_style;   // ident mapping
      }
      if ((xmask & GCFillRule)) {
         mask |= kGCFillRule;
         gval.fFillRule = xgval.fill_rule;     // ident mapping
      }
      if ((xmask & GCTile)) {
         mask |= kGCTile;
         gval.fTile = (Pixmap_t) xgval.tile;
      }
      if ((xmask & GCStipple)) {
         mask |= kGCStipple;
         gval.fStipple = (Pixmap_t) xgval.stipple;
      }
      if ((xmask & GCTileStipXOrigin)) {
         mask |= kGCTileStipXOrigin;
         gval.fTsXOrigin = xgval.ts_x_origin;
      }
      if ((xmask & GCTileStipYOrigin)) {
         mask |= kGCTileStipYOrigin;
         gval.fTsYOrigin = xgval.ts_y_origin;
      }
      if ((xmask & GCFont)) {
         mask |= kGCFont;
         gval.fFont = (FontH_t) xgval.font;
      }
      if ((xmask & GCSubwindowMode)) {
         mask |= kGCSubwindowMode;
         gval.fSubwindowMode = xgval.subwindow_mode;  // ident mapping
      }
      if ((xmask & GCGraphicsExposures)) {
         mask |= kGCGraphicsExposures;
         gval.fGraphicsExposures = (Bool_t) xgval.graphics_exposures;
      }
      if ((xmask & GCClipXOrigin)) {
         mask |= kGCClipXOrigin;
         gval.fClipXOrigin = xgval.clip_x_origin;
      }
      if ((xmask & GCClipYOrigin)) {
         mask |= kGCClipYOrigin;
         gval.fClipYOrigin = xgval.clip_y_origin;
      }
      if ((xmask & GCClipMask)) {
         mask |= kGCClipMask;
         gval.fClipMask = (Pixmap_t) xgval.clip_mask;
      }
      if ((xmask & GCDashOffset)) {
         mask |= kGCDashOffset;
         gval.fDashOffset = xgval.dash_offset;
      }
      if ((xmask & GCDashList)) {
         mask |= kGCDashList;
         gval.fDashes[0] = xgval.dashes;
         gval.fDashLen   = 1;
      }
      if ((xmask & GCArcMode)) {
         mask |= kGCArcMode;
         gval.fArcMode = xgval.arc_mode;   // ident mapping
      }
      gval.fMask = mask;
   }
}

//______________________________________________________________________________
void TGX11::GetWindowAttributes(Window_t id, WindowAttributes_t &attr)
{
   // Get window attributes and return filled in attributes structure.

   if (!id) return;

   XWindowAttributes xattr;

   XGetWindowAttributes((Display*)fDisplay, id, &xattr);

   attr.fX                  = xattr.x;
   attr.fY                  = xattr.y;
   attr.fWidth              = xattr.width;
   attr.fHeight             = xattr.height;
   attr.fBorderWidth        = xattr.border_width;
   attr.fDepth              = xattr.depth;
   attr.fVisual             = xattr.visual;
   attr.fRoot               = (Window_t) xattr.root;
   if (xattr.c_class == InputOutput) attr.fClass = kInputOutput;
   if (xattr.c_class == InputOnly)   attr.fClass = kInputOnly;
   attr.fBitGravity         = xattr.bit_gravity;  // assume ident mapping (rdm)
   attr.fWinGravity         = xattr.win_gravity;  // assume ident mapping (rdm)
   if (xattr.backing_store == NotUseful)  attr.fBackingStore = kNotUseful;
   if (xattr.backing_store == WhenMapped) attr.fBackingStore = kWhenMapped;
   if (xattr.backing_store == Always)     attr.fBackingStore = kAlways;
   attr.fBackingPlanes      = xattr.backing_planes;
   attr.fBackingPixel       = xattr.backing_pixel;
   attr.fSaveUnder          = (Bool_t) xattr.save_under;
   if ((Window) id == fRootWin)
      attr.fColormap        = (Colormap_t) fColormap;
   else
      attr.fColormap        = (Colormap_t) xattr.colormap;
   attr.fMapInstalled       = (Bool_t) xattr.map_installed;
   attr.fMapState           = xattr.map_state;         // ident mapping
   attr.fAllEventMasks      = xattr.all_event_masks;   // not ident, but not used by GUI classes
   attr.fYourEventMask      = xattr.your_event_mask;   // not ident, but not used by GUI classes
   attr.fDoNotPropagateMask = xattr.do_not_propagate_mask;
   attr.fOverrideRedirect   = (Bool_t) xattr.override_redirect;
   attr.fScreen             = xattr.screen;
}

//______________________________________________________________________________
Int_t TGX11::OpenDisplay(const char *dpyName)
{
   // Open connection to display server (if such a thing exist on the
   // current platform). On X11 this method returns on success the X
   // display socket descriptor (> 0), 0 in case of batch mode and < 0
   // in case of failure (cannot connect to display dpyName). It also
   // initializes the TGX11 class via Init(). Called from TGClient ctor.

#ifdef _REENTRANT
   // In some cases there can be problems due to XInitThreads, like when
   // using Qt, so we allow for it to be turned off
   if (gEnv->GetValue("X11.XInitThread", 1)) {
      // Must be very first call before any X11 call !!
      if (!XInitThreads())
         Warning("OpenDisplay", "system has no X11 thread support");
   }
#endif

   Display *dpy;
   if (!(dpy = XOpenDisplay(dpyName)))
      return -1;

   // Set custom X11 error handlers
   XSetErrorHandler(RootX11ErrorHandler);
   XSetIOErrorHandler(RootX11IOErrorHandler);

   if (gEnv->GetValue("X11.Sync", 0))
      XSynchronize(dpy, 1);

   // Init the GX11 class, sets a.o. fDisplay.
   if (!Init(dpy))
      return -1;

   return ConnectionNumber(dpy);
}

//______________________________________________________________________________
void TGX11::CloseDisplay()
{
   // Close connection to display server.

   XCloseDisplay((Display*)fDisplay);
   fDisplay = 0;
}

//______________________________________________________________________________
Display_t TGX11::GetDisplay() const
{
   // Returns handle to display (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   return (Display_t) fDisplay;
}

//______________________________________________________________________________
Visual_t TGX11::GetVisual() const
{
   // Returns handle to visual (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   return (Visual_t) fVisual;
}

//______________________________________________________________________________
Colormap_t TGX11::GetColormap() const
{
   // Returns handle to colormap (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   return (Colormap_t) fColormap;
}

//______________________________________________________________________________
Int_t TGX11::GetScreen() const
{
   // Returns screen number (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL
   // interface).

   return fScreenNumber;
}

//______________________________________________________________________________
Int_t TGX11::GetDepth() const
{
   // Returns depth of screen (number of bit planes). Equivalent to
   // GetPlanes().

   return fDepth;
}

//______________________________________________________________________________
Atom_t TGX11::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   Atom a = XInternAtom((Display*)fDisplay, (char *)atom_name, (Bool)only_if_exist);

   if (a == None) return kNone;
   return (Atom_t) a;
}

//______________________________________________________________________________
Window_t TGX11::GetDefaultRootWindow() const
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().

   return (Window_t) fRootWin;
}

//______________________________________________________________________________
Window_t TGX11::GetParent(Window_t id) const
{
   // Return the parent of the window.

   if (!id) return (Window_t)0;

   Window  root, parent;
   Window *children = 0;
   UInt_t  nchildren;

   XQueryTree((Display*)fDisplay, (Window) id, &root, &parent, &children, &nchildren);

   if (children) XFree(children);

   return (Window_t) parent;
}


//______________________________________________________________________________
FontStruct_t TGX11::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise an opaque pointer to the FontStruct_t.
   // Free the loaded font using DeleteFont().

   XFontStruct *fs = XLoadQueryFont((Display*)fDisplay, (char *)font_name);
   return (FontStruct_t) fs;
}

//______________________________________________________________________________
FontH_t TGX11::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.

   if (fs) {
      XFontStruct *fss = (XFontStruct *)fs;
      return fss->fid;
   }
   return 0;
}

//______________________________________________________________________________
void TGX11::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure obtained with LoadQueryFont().

   if (fDisplay) XFreeFont((Display*)fDisplay, (XFontStruct *) fs);
}

//______________________________________________________________________________
GContext_t TGX11::CreateGC(Drawable_t id, GCValues_t *gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).

   RXGCValues xgval;
   ULong_t   xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, xgval);

   if (!id || ((Drawable) id == fRootWin))
      id = (Drawable_t) fVisRootWin;

   GC gc = XCreateGC((Display*)fDisplay, (Drawable) id, xmask, &xgval);

   if (gval->fMask & kGCFont) MapGCFont((GContext_t)gc, gval->fFont);

   return (GContext_t) gc;
}

//______________________________________________________________________________
void TGX11::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.

   RXGCValues xgval;
   ULong_t   xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, xgval);

   XChangeGC((Display*)fDisplay, (GC) gc, xmask, &xgval);

   if (gval->fMask & kGCFont) MapGCFont((GContext_t)gc, gval->fFont);
}

//______________________________________________________________________________
void TGX11::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. If mask = 0 then copy all fields. Both org and
   // dest must exist.

   GCValues_t gval;
   RXGCValues  xgval;
   ULong_t    xmask;

   if (!mask) {
      // in this case copy all fields
      mask = kMaxUInt;
   }

   gval.fMask = mask;  // only set fMask used to convert to xmask
   MapGCValues(gval, xmask, xgval);

   XCopyGC((Display*)fDisplay, (GC) org, xmask, (GC) dest);
}

//______________________________________________________________________________
void TGX11::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.

   // Protection against deletion of global TGGC objects, which are
   // destructed after fDisplay has been closed.
   if (fDisplay)
      XFreeGC((Display*)fDisplay, (GC) gc);
}

//______________________________________________________________________________
Cursor_t TGX11::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).

   return (Cursor_t) fCursors[cursor];
}

//______________________________________________________________________________
void TGX11::SetCursor(Window_t id, Cursor_t curid)
{
   // Set the specified cursor.

   if (!id) return;

   XDefineCursor((Display*)fDisplay, (Window) id, (Cursor) curid);
}

//______________________________________________________________________________
Pixmap_t TGX11::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.

   return (Pixmap_t) XCreatePixmap((Display*)fDisplay, (Drawable) (id ? id : fRootWin), w, h, fDepth);
}

//______________________________________________________________________________
Pixmap_t TGX11::CreatePixmap(Drawable_t id, const char *bitmap,
            UInt_t width, UInt_t height, ULong_t forecolor, ULong_t backcolor,
            Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.

   return (Pixmap_t) XCreatePixmapFromBitmapData((Display*)fDisplay, (id ? id : fRootWin), (char *)bitmap,
                           width, height, forecolor, backcolor, depth);
}

//______________________________________________________________________________
Pixmap_t TGX11::CreateBitmap(Drawable_t id, const char *bitmap,
                             UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

   return (Pixmap_t) XCreateBitmapFromData((Display*)fDisplay, (id ? id : fRootWin), (char *)bitmap,
                                           width, height);
}

//______________________________________________________________________________
void TGX11::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.

   if (fDisplay) XFreePixmap((Display*)fDisplay, (Pixmap) pmap);
}

//______________________________________________________________________________
void TGX11::MapPictureAttributes(PictureAttributes_t &attr, RXpmAttributes &xpmattr,
                                 Bool_t toxpm)
{
   // Map a PictureAttributes_t to a XpmAttributes structure. If toxpm is
   // kTRUE map from attr to xpmattr, else map the other way.

#ifdef XpmVersion
   if (toxpm) {
      Mask_t  mask = attr.fMask;
      ULong_t xmask = 0;

      if ((mask & kPAColormap)) {
         xmask |= XpmColormap;
         xpmattr.colormap = (Colormap)attr.fColormap;
      }
      if ((mask & kPADepth)) {
         xmask |= XpmDepth;
         xpmattr.depth = attr.fDepth;
      }
      if ((mask & kPASize)) {
         xmask |= XpmSize;
         xpmattr.width  = attr.fWidth;
         xpmattr.height = attr.fHeight;
      }
      if ((mask & kPAHotspot)) {
         xmask |= XpmHotspot;
         xpmattr.x_hotspot = attr.fXHotspot;
         xpmattr.y_hotspot = attr.fYHotspot;
      }
      if ((mask & kPAReturnPixels)) {
         xmask |= XpmReturnPixels;
         xpmattr.pixels  = 0;  // output parameters
         xpmattr.npixels = 0;
      }
      if ((mask & kPACloseness)) {
         xmask |= XpmCloseness;
         xpmattr.closeness = attr.fCloseness;
      }
      xpmattr.valuemask = xmask;
   } else {
      ULong_t xmask = xpmattr.valuemask;
      Mask_t  mask  = 0;

      attr.fPixels  = 0;
      attr.fNpixels = 0;

      if ((xmask & XpmColormap)) {
         mask |= kPAColormap;
         attr.fColormap = (Colormap_t)xpmattr.colormap;
         mask |= kPADepth;
         attr.fDepth = xpmattr.depth;
      }
      if ((xmask & XpmSize)) {
         mask |= kPASize;
         attr.fWidth  = xpmattr.width;
         attr.fHeight = xpmattr.height;
      }
      if ((xmask & XpmHotspot)) {
         mask |= kPAHotspot;
         attr.fXHotspot = xpmattr.x_hotspot;
         attr.fYHotspot = xpmattr.y_hotspot;
      }
      if ((xmask & XpmReturnPixels)) {
         mask |= kPAReturnPixels;
         if (xpmattr.npixels) {
            attr.fPixels = new ULong_t[xpmattr.npixels];
            for (UInt_t i = 0; i < xpmattr.npixels; i++)
               attr.fPixels[i] = xpmattr.pixels[i];
            attr.fNpixels = xpmattr.npixels;
         }
      }
      if ((xmask & XpmCloseness)) {
         mask |= kPACloseness;
         attr.fCloseness = xpmattr.closeness;
      }
      attr.fMask = mask;
   }
#endif
}

//______________________________________________________________________________
Bool_t TGX11::CreatePictureFromFile(Drawable_t id, const char *filename,
                                    Pixmap_t &pict, Pixmap_t &pict_mask,
                                    PictureAttributes_t &attr)
{
   // Create a picture pixmap from data on file. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

   if (strstr(filename, ".gif") || strstr(filename, ".GIF")) {
      pict = ReadGIF(0, 0, filename, id);
      pict_mask = kNone;
      attr.fDepth = fDepth;
      Int_t dummy;
      GetWindowSize(pict, dummy, dummy, attr.fWidth, attr.fHeight);
      return kTRUE;
   }

#ifdef XpmVersion
   RXpmAttributes xpmattr;

   MapPictureAttributes(attr, xpmattr);

   // make sure pixel depth of pixmap is the same as in the visual
   if ((Drawable) id == fRootWin && fRootWin != fVisRootWin) {
      xpmattr.valuemask |= XpmDepth;
      xpmattr.depth = fDepth;
   }

   Int_t res = XpmReadFileToPixmap((Display*)fDisplay, (id ? id : fRootWin), (char*)filename,
                                   (Pixmap*)&pict, (Pixmap*)&pict_mask, &xpmattr);

   MapPictureAttributes(attr, xpmattr, kFALSE);
   XpmFreeAttributes(&xpmattr);

   if (res == XpmSuccess || res == XpmColorError)
      return kTRUE;

   if (pict) {
      XFreePixmap((Display*)fDisplay, (Pixmap)pict);
      pict = kNone;
   }
   if (pict_mask) {
      XFreePixmap((Display*)fDisplay, (Pixmap)pict_mask);
      pict_mask = kNone;
   }
#else
   Error("CreatePictureFromFile", "cannot get picture, not compiled with Xpm");
#endif

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGX11::CreatePictureFromData(Drawable_t id, char **data, Pixmap_t &pict,
                                    Pixmap_t &pict_mask, PictureAttributes_t &attr)
{
   // Create a pixture pixmap from data. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

#ifdef XpmVersion
   RXpmAttributes xpmattr;

   MapPictureAttributes(attr, xpmattr);

   // make sure pixel depth of pixmap is the same as in the visual
   if ((Drawable) id == fRootWin && fRootWin != fVisRootWin) {
      xpmattr.valuemask |= XpmDepth;
      xpmattr.depth = fDepth;
   }

   Int_t res = XpmCreatePixmapFromData((Display*)fDisplay, (id ? id : fRootWin), data, (Pixmap*)&pict,
                                       (Pixmap*)&pict_mask, &xpmattr);

   MapPictureAttributes(attr, xpmattr, kFALSE);
   XpmFreeAttributes(&xpmattr);

   if (res == XpmSuccess || res == XpmColorError)
      return kTRUE;

   if (pict) {
      XFreePixmap((Display*)fDisplay, (Pixmap)pict);
      pict = kNone;
   }
   if (pict_mask) {
      XFreePixmap((Display*)fDisplay, (Pixmap)pict_mask);
      pict_mask = kNone;
   }
#else
   Error("CreatePictureFromData", "cannot get picture, not compiled with Xpm");
#endif

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGX11::ReadPictureDataFromFile(const char *filename, char ***ret_data)
{
   // Read picture data from file and store in ret_data. Returns kTRUE in
   // case of success, kFALSE otherwise.

#ifdef XpmVersion
   if (XpmReadFileToData((char*)filename, ret_data) == XpmSuccess)
      return kTRUE;
#else
   Error("ReadPictureFromDataFile", "cannot get picture, not compiled with Xpm");
#endif
   return kFALSE;
}

//______________________________________________________________________________
void TGX11::DeletePictureData(void *data)
{
   // Delete picture data created by the function ReadPictureDataFromFile.

#ifdef XpmVersion
   // some older libXpm's don't have this function and it is typically
   // implemented with a simple free()
   // XpmFree(data);
   free(data);
#endif
}

//______________________________________________________________________________
void TGX11::SetDashes(GContext_t gc, Int_t offset, const char *dash_list, Int_t n)
{
   // Specify a dash pattertn. Offset defines the phase of the pattern.
   // Each element in the dash_list array specifies the length (in pixels)
   // of a segment of the pattern. N defines the length of the list.

   XSetDashes((Display*)fDisplay, (GC) gc, offset, (char *)dash_list, n);
}

//______________________________________________________________________________
void TGX11::MapColorStruct(ColorStruct_t *color, RXColor &xcolor)
{
   // Map a ColorStruct_t to a XColor structure.

   xcolor.pixel = color->fPixel;
   xcolor.red   = color->fRed;
   xcolor.green = color->fGreen;
   xcolor.blue  = color->fBlue;
   xcolor.flags = color->fMask;  //ident mapping
}

//______________________________________________________________________________
Bool_t TGX11::ParseColor(Colormap_t cmap, const char *cname, ColorStruct_t &color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   XColor xc;

   if (XParseColor((Display*)fDisplay, (Colormap)cmap, (char *)cname, &xc)) {
      color.fPixel = 0;
      color.fRed   = xc.red;
      color.fGreen = xc.green;
      color.fBlue  = xc.blue;
      color.fMask  = kDoRed | kDoGreen | kDoBlue;
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGX11::AllocColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   RXColor xc;

   MapColorStruct(&color, xc);

   color.fPixel = 0;
   if (AllocColor((Colormap)cmap, &xc)) {
      color.fPixel = xc.pixel;
      return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
void TGX11::QueryColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   XColor xc;

   xc.pixel = color.fPixel;

   // still very slight dark shift ??
   //QueryColors((Colormap)cmap, &xc, 1);
   //printf("1 xc.red = %u, xc.green = %u, xc.blue = %u\n", xc.red, xc.green, xc.blue);
   XQueryColor((Display*)fDisplay, (Colormap)cmap, &xc);
   //printf("2 xc.red = %u, xc.green = %u, xc.blue = %u\n", xc.red, xc.green, xc.blue);

   color.fRed   = xc.red;
   color.fGreen = xc.green;
   color.fBlue  = xc.blue;
}

//______________________________________________________________________________
void TGX11::FreeColor(Colormap_t cmap, ULong_t pixel)
{
   // Free color cell with specified pixel value.

   if (fRedDiv == -1)
      XFreeColors((Display*)fDisplay, (Colormap)cmap, &pixel, 1, 0);
}

//______________________________________________________________________________
Int_t TGX11::EventsPending()
{
   // Returns number of pending events.

   if (!fDisplay) return 0;
   return XPending((Display*)fDisplay);
}

//______________________________________________________________________________
void TGX11::NextEvent(Event_t &event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   XNextEvent((Display*)fDisplay, (XEvent*)fXEvent);

   // fill in Event_t
   MapEvent(event, fXEvent, kFALSE);
}

//______________________________________________________________________________
void TGX11::MapModifierState(UInt_t &state, UInt_t &xstate, Bool_t tox)
{
   // Map modifier key state to or from X.

   if (tox) {
      xstate = 0;
      if ((state & kKeyShiftMask))
         xstate |= ShiftMask;
      if ((state & kKeyLockMask))
         xstate |= LockMask;
      if ((state & kKeyControlMask))
         xstate |= ControlMask;
      if ((state & kKeyMod1Mask))
         xstate |= Mod1Mask;
      if ((state & kKeyMod2Mask))
         xstate |= Mod2Mask;
      if ((state & kKeyMod3Mask))
         xstate |= Mod3Mask;
      if ((state & kKeyMod4Mask))
         xstate |= Mod4Mask;
      if ((state & kKeyMod5Mask))
         xstate |= Mod5Mask;
      if ((state & kButton1Mask))
         xstate |= Button1Mask;
      if ((state & kButton2Mask))
         xstate |= Button2Mask;
      if ((state & kButton3Mask))
         xstate |= Button3Mask;
      if ((state & kAnyModifier))
         xstate |= AnyModifier;      // or should it be = instead of |= ?
   } else {
      state = 0;
      if ((xstate & ShiftMask))
         state |= kKeyShiftMask;
      if ((xstate & LockMask))
         state |= kKeyLockMask;
      if ((xstate & ControlMask))
         state |= kKeyControlMask;
      if ((xstate & Mod1Mask))
         state |= kKeyMod1Mask;
      if ((xstate & Mod2Mask))
         state |= kKeyMod2Mask;
      if ((xstate & Mod3Mask))
         state |= kKeyMod3Mask;
      if ((xstate & Mod4Mask))
         state |= kKeyMod4Mask;
      if ((xstate & Mod5Mask))
         state |= kKeyMod5Mask;
      if ((xstate & Button1Mask))
         state |= kButton1Mask;
      if ((xstate & Button2Mask))
         state |= kButton2Mask;
      if ((xstate & Button3Mask))
         state |= kButton3Mask;
      if ((xstate & AnyModifier))
         state |= kAnyModifier;      // idem
   }
}

//______________________________________________________________________________
void TGX11::MapEvent(Event_t &ev, void *xevi, Bool_t tox)
{
   // Map Event_t structure to XEvent structure. If tox is false
   // map the other way.

   XEvent &xev = *(XEvent *)xevi;

   if (tox) {
      // map from Event_t to XEvent
      xev.type = 0;
      if (ev.fType == kGKeyPress)        xev.type = KeyPress;
      if (ev.fType == kKeyRelease)       xev.type = KeyRelease;
      if (ev.fType == kButtonPress)      xev.type = ButtonPress;
      if (ev.fType == kButtonRelease)    xev.type = ButtonRelease;
      if (ev.fType == kMotionNotify)     xev.type = MotionNotify;
      if (ev.fType == kEnterNotify)      xev.type = EnterNotify;
      if (ev.fType == kLeaveNotify)      xev.type = LeaveNotify;
      if (ev.fType == kFocusIn)          xev.type = FocusIn;
      if (ev.fType == kFocusOut)         xev.type = FocusOut;
      if (ev.fType == kExpose)           xev.type = Expose;
      if (ev.fType == kConfigureNotify)  xev.type = ConfigureNotify;
      if (ev.fType == kMapNotify)        xev.type = MapNotify;
      if (ev.fType == kUnmapNotify)      xev.type = UnmapNotify;
      if (ev.fType == kDestroyNotify)    xev.type = DestroyNotify;
      if (ev.fType == kClientMessage)    xev.type = ClientMessage;
      if (ev.fType == kSelectionClear)   xev.type = SelectionClear;
      if (ev.fType == kSelectionRequest) xev.type = SelectionRequest;
      if (ev.fType == kSelectionNotify)  xev.type = SelectionNotify;
      if (ev.fType == kColormapNotify)   xev.type = ColormapNotify;

      xev.xany.window     = (Window) ev.fWindow;
      xev.xany.send_event = (Bool) ev.fSendEvent;
      xev.xany.display    = (Display*)fDisplay;

      if (ev.fType == kGKeyPress || ev.fType == kKeyRelease) {
         xev.xkey.time   = (Time) ev.fTime;
         xev.xkey.x      = ev.fX;
         xev.xkey.y      = ev.fY;
         xev.xkey.x_root = ev.fXRoot;
         xev.xkey.y_root = ev.fYRoot;
         MapModifierState(ev.fState, xev.xkey.state, kTRUE); // key mask
         xev.xkey.keycode = ev.fCode;    // key code
      }
      if (ev.fType == kSelectionNotify) {
         xev.xselection.time      = (Time) ev.fTime;
         xev.xselection.requestor = (Window) ev.fUser[0];
         xev.xselection.selection = (Atom) ev.fUser[1];
         xev.xselection.target    = (Atom) ev.fUser[2];
         xev.xselection.property  = (Atom) ev.fUser[3];
      }
      if (ev.fType == kClientMessage) {
         xev.xclient.message_type = ev.fHandle;
         xev.xclient.format       = ev.fFormat;
         xev.xclient.data.l[0]    = ev.fUser[0];
         if (sizeof(ev.fUser[0]) > 4) {
            SplitLong(ev.fUser[1], xev.xclient.data.l[1], xev.xclient.data.l[3]);
            SplitLong(ev.fUser[2], xev.xclient.data.l[2], xev.xclient.data.l[4]);
         } else {
            xev.xclient.data.l[1]    = ev.fUser[1];
            xev.xclient.data.l[2]    = ev.fUser[2];
            xev.xclient.data.l[3]    = ev.fUser[3];
            xev.xclient.data.l[4]    = ev.fUser[4];
         }
      }
   } else {
      // map from XEvent to Event_t
      ev.fType = kOtherEvent;
      if (xev.type == KeyPress)         ev.fType = kGKeyPress;
      if (xev.type == KeyRelease)       ev.fType = kKeyRelease;
      if (xev.type == ButtonPress)      ev.fType = kButtonPress;
      if (xev.type == ButtonRelease)    ev.fType = kButtonRelease;
      if (xev.type == MotionNotify)     ev.fType = kMotionNotify;
      if (xev.type == EnterNotify)      ev.fType = kEnterNotify;
      if (xev.type == LeaveNotify)      ev.fType = kLeaveNotify;
      if (xev.type == FocusIn)          ev.fType = kFocusIn;
      if (xev.type == FocusOut)         ev.fType = kFocusOut;
      if (xev.type == Expose)           ev.fType = kExpose;
      if (xev.type == GraphicsExpose)   ev.fType = kExpose;
      if (xev.type == ConfigureNotify)  ev.fType = kConfigureNotify;
      if (xev.type == MapNotify)        ev.fType = kMapNotify;
      if (xev.type == UnmapNotify)      ev.fType = kUnmapNotify;
      if (xev.type == DestroyNotify)    ev.fType = kDestroyNotify;
      if (xev.type == ClientMessage)    ev.fType = kClientMessage;
      if (xev.type == SelectionClear)   ev.fType = kSelectionClear;
      if (xev.type == SelectionRequest) ev.fType = kSelectionRequest;
      if (xev.type == SelectionNotify)  ev.fType = kSelectionNotify;
      if (xev.type == ColormapNotify)   ev.fType = kColormapNotify;

      ev.fWindow    = (Window_t) xev.xany.window;
      ev.fSendEvent = xev.xany.send_event ? kTRUE : kFALSE;

      if (ev.fType == kGKeyPress || ev.fType == kKeyRelease) {
         ev.fTime      = (Time_t) xev.xkey.time;
         ev.fX         = xev.xkey.x;
         ev.fY         = xev.xkey.y;
         ev.fXRoot     = xev.xkey.x_root;
         ev.fYRoot     = xev.xkey.y_root;
         MapModifierState(ev.fState, xev.xkey.state, kFALSE); // key mask
         ev.fCode      = xev.xkey.keycode;    // key code
         ev.fUser[0]   = xev.xkey.subwindow;  // child window
      }
      if (ev.fType == kButtonPress || ev.fType == kButtonRelease) {
         ev.fTime      = (Time_t) xev.xbutton.time;
         ev.fX         = xev.xbutton.x;
         ev.fY         = xev.xbutton.y;
         ev.fXRoot     = xev.xbutton.x_root;
         ev.fYRoot     = xev.xbutton.y_root;
         MapModifierState(ev.fState, xev.xbutton.state, kFALSE); // button mask
         ev.fCode      = xev.xbutton.button;    // button code
         ev.fUser[0]   = xev.xbutton.subwindow; // child window
      }
      if (ev.fType == kMotionNotify) {
         ev.fTime      = (Time_t) xev.xmotion.time;
         ev.fX         = xev.xmotion.x;
         ev.fY         = xev.xmotion.y;
         ev.fXRoot     = xev.xmotion.x_root;
         ev.fYRoot     = xev.xmotion.y_root;
         MapModifierState(ev.fState, xev.xmotion.state, kFALSE); // key or button mask
         ev.fUser[0]   = xev.xmotion.subwindow; // child window
      }
      if (ev.fType == kEnterNotify || ev.fType == kLeaveNotify) {
         ev.fTime      = (Time_t) xev.xcrossing.time;
         ev.fX         = xev.xcrossing.x;
         ev.fY         = xev.xcrossing.y;
         ev.fXRoot     = xev.xcrossing.x_root;
         ev.fYRoot     = xev.xcrossing.y_root;
         ev.fCode      = xev.xcrossing.mode; // NotifyNormal, NotifyGrab, NotifyUngrab
         MapModifierState(ev.fState, xev.xcrossing.state, kFALSE); // key or button mask
      }
      if (ev.fType == kFocusIn || ev.fType == kFocusOut) {
         // check this when porting to Win32 (see also TGTextEntry::HandleFocusChange)
         ev.fCode      = xev.xfocus.mode; // NotifyNormal, NotifyGrab, NotifyUngrab
         ev.fState     = xev.xfocus.detail; // NotifyPointer et al.
      }
      if (ev.fType == kExpose) {
         ev.fX         = xev.xexpose.x;
         ev.fY         = xev.xexpose.y;
         ev.fWidth     = xev.xexpose.width;   // width and
         ev.fHeight    = xev.xexpose.height;  // height of exposed area
         ev.fCount     = xev.xexpose.count;   // number of expose events still to come
      }
      if (ev.fType == kConfigureNotify) {
         ev.fX         = xev.xconfigure.x;
         ev.fY         = xev.xconfigure.y;
         ev.fWidth     = xev.xconfigure.width;
         ev.fHeight    = xev.xconfigure.height;
      }
      if (ev.fType == kMapNotify || ev.fType == kUnmapNotify) {
         ev.fHandle = xev.xmap.window;  // window to be (un)mapped
      }
      if (ev.fType == kDestroyNotify) {
         ev.fHandle = xev.xdestroywindow.window;  // window to be destroyed
      }
      if (ev.fType == kClientMessage) {
         ev.fHandle  = xev.xclient.message_type;
         ev.fFormat  = xev.xclient.format;
         ev.fUser[0] = xev.xclient.data.l[0];
         if (sizeof(ev.fUser[0]) > 4) {
            AsmLong(xev.xclient.data.l[1], xev.xclient.data.l[3], ev.fUser[1]);
            AsmLong(xev.xclient.data.l[2], xev.xclient.data.l[4], ev.fUser[2]);
         } else {
            ev.fUser[1] = xev.xclient.data.l[1];
            ev.fUser[2] = xev.xclient.data.l[2];
            ev.fUser[3] = xev.xclient.data.l[3];
            ev.fUser[4] = xev.xclient.data.l[4];
         }
      }
      if (ev.fType == kSelectionClear) {
         ev.fUser[0] = xev.xselectionclear.selection;
      }
      if (ev.fType == kSelectionRequest) {
         ev.fTime    = (Time_t) xev.xselectionrequest.time;
         ev.fUser[0] = xev.xselectionrequest.requestor;
         ev.fUser[1] = xev.xselectionrequest.selection;
         ev.fUser[2] = xev.xselectionrequest.target;
         ev.fUser[3] = xev.xselectionrequest.property;
      }
      if (ev.fType == kSelectionNotify) {
         ev.fTime    = (Time_t) xev.xselection.time;
         ev.fUser[0] = xev.xselection.requestor;
         ev.fUser[1] = xev.xselection.selection;
         ev.fUser[2] = xev.xselection.target;
         ev.fUser[3] = xev.xselection.property;
      }
      if (ev.fType == kColormapNotify) {
         ev.fHandle = xev.xcolormap.colormap;
         ev.fCode   = xev.xcolormap.state; // ColormapUninstalled, ColormapInstalled
         ev.fState  = xev.xcolormap.c_new; // true if new colormap
      }
   }
}

//______________________________________________________________________________
void TGX11::Bell(Int_t percent)
{
   // Sound bell. Percent is loudness from -100% .. 100%.

   XBell((Display*)fDisplay, percent);
}

//______________________________________________________________________________
void TGX11::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                     Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
                     Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.

   if (!src || !dest) return;

   XCopyArea((Display*)fDisplay, src, dest, (GC) gc, src_x, src_y, width, height,
             dest_x, dest_y);
}

//______________________________________________________________________________
void TGX11::ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr)
{
   // Change window attributes.

   if (!id) return;

   RXSetWindowAttributes xattr;
   ULong_t              xmask = 0;

   if (attr)
      MapSetWindowAttributes(attr, xmask, xattr);

   XChangeWindowAttributes((Display*)fDisplay, (Window) id, xmask, &xattr);

   if (attr && (attr->fMask & kWABorderWidth))
      XSetWindowBorderWidth((Display*)fDisplay, (Window) id, attr->fBorderWidth);
}

//______________________________________________________________________________
void TGX11::ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                           UChar_t *data, Int_t len)
{
   // This function alters the property for the specified window and
   // causes the X server to generate a PropertyNotify event on that
   // window.

   if (!id) return;

   XChangeProperty((Display*)fDisplay, (Window) id, (Atom) property, (Atom) type,
                   8, PropModeReplace, data, len);
}

//______________________________________________________________________________
void TGX11::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.

   if (!id) return;

   XDrawLine((Display*)fDisplay, (Drawable) id, (GC) gc, x1, y1, x2, y2);
}

//______________________________________________________________________________
void TGX11::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.

   if (!id) return;

   XClearArea((Display*)fDisplay, (Window) id, x, y, w, h, False);
}

//______________________________________________________________________________
Bool_t TGX11::CheckEvent(Window_t id, EGEventType type, Event_t &ev)
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.

   if (!id) return kFALSE;

   Event_t tev;
   XEvent  xev;

   tev.fCode = 0;
   tev.fState = 0;
   tev.fWindow = 0;
   tev.fUser[0] = tev.fUser[1] = tev.fUser[2] = tev.fUser[3] = tev.fUser[4] = 0;
   tev.fCount = 0;
   tev.fFormat = 0;
   tev.fHandle = 0;
   tev.fSendEvent = 0;
   tev.fTime = 0;
   tev.fX = tev.fY = 0;
   tev.fXRoot = tev.fYRoot = 0;
   tev.fType = type;
   MapEvent(tev, &xev);

   Bool r = XCheckTypedWindowEvent((Display*)fDisplay, (Window) id, xev.type, &xev);

   if (r) MapEvent(ev, &xev, kFALSE);

   return r ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGX11::SendEvent(Window_t id, Event_t *ev)
{
   // Send event ev to window id.

   if (!ev || !id) return;

   XEvent xev;

   MapEvent(*ev, &xev);

   XSendEvent((Display*)fDisplay, (Window) id, False, None, &xev);
}

//______________________________________________________________________________
void TGX11::WMDeleteNotify(Window_t id)
{
   // Tell WM to send message when window is closed via WM.

   if (!id) return;

   XSetWMProtocols((Display*)fDisplay, (Window) id, &gWM_DELETE_WINDOW, 1);
}

//______________________________________________________________________________
void TGX11::SetKeyAutoRepeat(Bool_t on)
{
   // Turn key auto repeat on or off.

   if (on)
      XAutoRepeatOn((Display*)fDisplay);
   else
      XAutoRepeatOff((Display*)fDisplay);
}

//______________________________________________________________________________
void TGX11::GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab)
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.

//   if (!id) return;

   UInt_t xmod;

   MapModifierState(modifier, xmod);

   if (grab)
      XGrabKey((Display*)fDisplay, keycode, xmod, (Window) id, True,
               GrabModeAsync, GrabModeAsync);
   else
      XUngrabKey((Display*)fDisplay, keycode, xmod, (Window) id);
}

//______________________________________________________________________________
void TGX11::GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                       UInt_t evmask, Window_t confine, Cursor_t cursor,
                       Bool_t grab)
{
   // Establish passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.

   if (!id) return;

   UInt_t xmod;

   MapModifierState(modifier, xmod);

   if (grab) {
      UInt_t xevmask;
      MapEventMask(evmask, xevmask);

      XGrabButton((Display*)fDisplay, button, xmod, (Window) id, True, xevmask,
                  GrabModeAsync, GrabModeAsync, (Window) confine,
                  (Cursor) cursor);
   } else
      XUngrabButton((Display*)fDisplay, button, xmod, (Window) id);
}

//______________________________________________________________________________
void TGX11::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                        Cursor_t cursor, Bool_t grab, Bool_t owner_events)
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.

//   if (!id) return;

   if (grab) {
      UInt_t xevmask;
      MapEventMask(evmask, xevmask);

      XGrabPointer((Display*)fDisplay, (Window) id, (Bool) owner_events,
                   xevmask, GrabModeAsync, GrabModeAsync, (Window) confine,
                   (Cursor) cursor, CurrentTime);
   } else
      XUngrabPointer((Display*)fDisplay, CurrentTime);
}

//______________________________________________________________________________
void TGX11::SetWindowName(Window_t id, char *name)
{
   // Set window name.

   if (!id) return;

   XTextProperty wname;

   if (XStringListToTextProperty(&name, 1, &wname) == 0) {
      Error("SetWindowName", "cannot allocate window name \"%s\"", name);
      return;
   }
   XSetWMName((Display*)fDisplay, (Window) id, &wname);
   XFree(wname.value);
}

//______________________________________________________________________________
void TGX11::SetIconName(Window_t id, char *name)
{
   // Set window icon name.

   if (!id) return;

   XTextProperty wname;

   if (XStringListToTextProperty(&name, 1, &wname) == 0) {
      Error("SetIconName", "cannot allocate icon name \"%s\"", name);
      return;
   }
   XSetWMIconName((Display*)fDisplay, (Window) id, &wname);
   XFree(wname.value);
}

//______________________________________________________________________________
void TGX11::SetIconPixmap(Window_t id, Pixmap_t pic)
{
   // Set pixmap the WM can use when the window is iconized.

   if (!id) return;

   XWMHints hints;

   hints.flags = IconPixmapHint;
   hints.icon_pixmap = (Pixmap) pic;

   XSetWMHints((Display*)fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetClassHints(Window_t id, char *className, char *resourceName)
{
   // Set the windows class and resource name.

   if (!id) return;

   XClassHint class_hints;

   class_hints.res_class = className;
   class_hints.res_name  = resourceName;
   XSetClassHint((Display*)fDisplay, (Window) id, &class_hints);
}

//______________________________________________________________________________
void TGX11::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).

   if (!id) return;

   MWMHintsProperty_t prop;

   prop.fDecorations = value;
   prop.fFunctions   = funcs;
   prop.fInputMode   = input;
   prop.fFlags       = kMWMHintsDecorations | kMWMHintsFunctions | kMWMHintsInputMode;

   XChangeProperty((Display*)fDisplay, (Window) id, gMOTIF_WM_HINTS, gMOTIF_WM_HINTS, 32,
                   PropModeReplace, (UChar_t *)&prop, kPropMWMHintElements);
}

//______________________________________________________________________________
void TGX11::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   // Tell the window manager the desired window position.

   if (!id) return;

   XSizeHints hints;

   hints.flags = USPosition | PPosition;
   hints.x = x;
   hints.y = y;

   XSetWMNormalHints((Display*)fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   // Tell the window manager the desired window size.

   if (!id) return;

   XSizeHints hints;

   hints.flags = USSize | PSize | PBaseSize;
   hints.width = hints.base_width = w;
   hints.height = hints.base_height = h;

   XSetWMNormalHints((Display*)fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                           UInt_t wmax, UInt_t hmax,
                           UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   if (!id) return;

   XSizeHints hints;

   hints.flags = PMinSize | PMaxSize | PResizeInc;
   hints.min_width   = (Int_t)wmin;
   hints.max_width   = (Int_t)wmax;
   hints.min_height  = (Int_t)hmin;
   hints.max_height  = (Int_t)hmax;
   hints.width_inc   = (Int_t)winc;
   hints.height_inc  = (Int_t)hinc;

   XSetWMNormalHints((Display*)fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMState(Window_t id, EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   if (!id) return;

   XWMHints hints;
   Int_t    xstate = NormalState;

   if (state == kNormalState)
      xstate = NormalState;
   if (state == kIconicState)
      xstate = IconicState;

   hints.flags = StateHint;
   hints.initial_state = xstate;

   XSetWMHints((Display*)fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMTransientHint(Window_t id, Window_t main_id)
{
   // Tell window manager that window is a transient window of main.

   if (!id) return;

   XSetTransientForHint((Display*)fDisplay, (Window) id, (Window) main_id);
}

//______________________________________________________________________________
void TGX11::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                       const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y).

   if (!id) return;

   XDrawString((Display*)fDisplay, (Drawable) id, (GC) gc, x, y, (char *) s, len);
}

//______________________________________________________________________________
Int_t TGX11::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return length of string in pixels. Size depends on font.

   return XTextWidth((XFontStruct*) font, (char*) s, len);
}

//______________________________________________________________________________
void TGX11::GetFontProperties(FontStruct_t font, Int_t &max_ascent, Int_t &max_descent)
{
   // Return some font properties.

   XFontStruct *f = (XFontStruct *) font;

   max_ascent  = f->max_bounds.ascent;
   max_descent = f->max_bounds.descent;
}

//______________________________________________________________________________
void TGX11::GetGCValues(GContext_t gc, GCValues_t &gval)
{
   // Get current values from graphics context gc. Which values of the
   // context to get is encoded in the GCValues::fMask member. If fMask = 0
   // then copy all fields.

   RXGCValues xgval;
   ULong_t   xmask;

   if (!gval.fMask) {
      // in this case copy all fields
      gval.fMask = kMaxUInt;
   }

   MapGCValues(gval, xmask, xgval);

   XGetGCValues((Display*)fDisplay, (GC) gc, xmask, &xgval);

   MapGCValues(gval, xmask, xgval, kFALSE);
}

//______________________________________________________________________________
FontStruct_t TGX11::GetFontStruct(FontH_t fh)
{
   // Retrieve associated font structure once we have the font handle.
   // Free returned FontStruct_t using FreeFontStruct().

   XFontStruct *fs;

   fs = XQueryFont((Display*)fDisplay, (Font) fh);

   return (FontStruct_t) fs;
}

//______________________________________________________________________________
void TGX11::FreeFontStruct(FontStruct_t fs)
{
   // Free font structure returned by GetFontStruct().

   // in XFree86 4.0 XFreeFontInfo() is broken, ok again in 4.0.1
   static int xfree86_400 = -1;
   if (xfree86_400 == -1) {
      if (strstr(XServerVendor((Display*)fDisplay), "XFree86") &&
          XVendorRelease((Display*)fDisplay) == 4000)
         xfree86_400 = 1;
      else
         xfree86_400 = 0;
   }

   if (xfree86_400 == 0)
      XFreeFontInfo(0, (XFontStruct *) fs, 1);
}

//______________________________________________________________________________
void TGX11::ClearWindow(Window_t id)
{
   // Clear window.

   if (!id) return;

   XClearWindow((Display*)fDisplay, (Window) id);
}

//______________________________________________________________________________
Int_t TGX11::KeysymToKeycode(UInt_t keysym)
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).

   UInt_t xkeysym;
   MapKeySym(keysym, xkeysym);

   return XKeysymToKeycode((Display*)fDisplay, xkeysym);
}

//______________________________________________________________________________
void TGX11::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a filled rectangle. Filling is done according to the gc.

   if (!id) return;

   XFillRectangle((Display*)fDisplay, (Drawable) id, (GC) gc, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a rectangle outline.

   if (!id) return;

   XDrawRectangle((Display*)fDisplay, (Drawable) id, (GC) gc, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg)
{
   // Draws multiple line segments. Each line is specified by a pair of points.

   if (!id) return;

   XDrawSegments((Display*)fDisplay, (Drawable) id, (GC) gc, (XSegment *) seg, nseg);
}

//______________________________________________________________________________
void TGX11::SelectInput(Window_t id, UInt_t evmask)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   if (!id) return;

   UInt_t xevmask;

   MapEventMask(evmask, xevmask);

   XSelectInput((Display*)fDisplay, (Window) id, xevmask);
}

//______________________________________________________________________________
Window_t TGX11::GetInputFocus()
{
   // Returns the window id of the window having the input focus.

   Window focus;
   int    return_to;

   XGetInputFocus((Display*)fDisplay, &focus, &return_to);
   return (Window_t) focus;
}

//______________________________________________________________________________
void TGX11::SetInputFocus(Window_t id)
{
   // Set keyboard input focus to window id.

   if (!id) return;

   XWindowAttributes xattr;

   XGetWindowAttributes((Display*)fDisplay, (Window) id, &xattr);

   if (xattr.map_state == IsViewable)
      XSetInputFocus((Display*)fDisplay, (Window) id, RevertToParent, CurrentTime);
}

//______________________________________________________________________________
Window_t TGX11::GetPrimarySelectionOwner()
{
   // Returns the window id of the current owner of the primary selection.
   // That is the window in which, for example some text is selected.

   return (Window_t) XGetSelectionOwner((Display*)fDisplay, XA_PRIMARY);
}

//______________________________________________________________________________
void TGX11::SetPrimarySelectionOwner(Window_t id)
{
   // Makes the window id the current owner of the primary selection.
   // That is the window in which, for example some text is selected.

   if (!id) return;

   XSetSelectionOwner((Display*)fDisplay, XA_PRIMARY, id, CurrentTime);
}

//______________________________________________________________________________
void TGX11::ConvertPrimarySelection(Window_t id, Atom_t clipboard, Time_t when)
{
   // XConvertSelection() causes a SelectionRequest event to be sent to the
   // current primary selection owner. This event specifies the selection
   // property (primary selection), the format into which to convert that
   // data before storing it (target = XA_STRING), the property in which
   // the owner will place the information (sel_property), the window that
   // wants the information (id), and the time of the conversion request
   // (when).
   // The selection owner responds by sending a SelectionNotify event, which
   // confirms the selected atom and type.

   if (!id) return;

   XConvertSelection((Display*)fDisplay, XA_PRIMARY, XA_STRING, (Atom) clipboard,
                     (Window) id, (Time) when);
}

//______________________________________________________________________________
void TGX11::LookupString(Event_t *event, char *buf, Int_t buflen, UInt_t &keysym)
{
   // Convert the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // keyboard mapping). In buf a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.

   XEvent xev;
   KeySym xkeysym;

   MapEvent(*event, &xev);

   int n = XLookupString(&xev.xkey, buf, buflen-1, &xkeysym, 0);
   if (n >= buflen)
      Error("LookupString", "buf too small, must be at least %d", n+1);
   else
      buf[n] = 0;

   UInt_t ks, xks = (UInt_t) xkeysym;
   MapKeySym(ks, xks, kFALSE);
   keysym = (Int_t) ks;
}

//______________________________________________________________________________
void TGX11::MapKeySym(UInt_t &keysym, UInt_t &xkeysym, Bool_t tox)
{
   // Map to and from X key symbols. Keysym are the values returned by
   // XLookUpString.

   if (tox) {
      xkeysym = XK_VoidSymbol;
      if (keysym < 127) {
         xkeysym = keysym;
      } else if (keysym >= kKey_F1 && keysym <= kKey_F35) {
         xkeysym = XK_F1 + (keysym - (UInt_t)kKey_F1);  // function keys
      } else {
         for (int i = 0; gKeyMap[i].fKeySym; i++) {    // any other keys
            if (keysym == (UInt_t) gKeyMap[i].fKeySym) {
               xkeysym = (UInt_t) gKeyMap[i].fXKeySym;
               break;
            }
         }
      }
   } else {
      keysym = kKey_Unknown;
      // commentary in X11/keysymdef says that X codes match ASCII
      if (xkeysym < 127) {
         keysym = xkeysym;
      } else if (xkeysym >= XK_F1 && xkeysym <= XK_F35) {
         keysym = kKey_F1 + (xkeysym - XK_F1);          // function keys
      } else if (xkeysym >= XK_KP_0 && xkeysym <= XK_KP_9) {
         keysym = kKey_0 + (xkeysym - XK_KP_0);         // numeric keypad keys
      } else {
         for (int i = 0; gKeyMap[i].fXKeySym; i++) {   // any other keys
            if (xkeysym == gKeyMap[i].fXKeySym) {
               keysym = (UInt_t) gKeyMap[i].fKeySym;
               break;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGX11::GetPasteBuffer(Window_t id, Atom_t atom, TString &text, Int_t &nchar,
                           Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.

   if (!id) return;

   Atom actual_type, property = (Atom) atom;
   int  actual_format;
   ULong_t nitems, bytes_after, nread;
   unsigned char *data;

   nchar = 0;
   text  = "";

   if (property == None) return;

   // get past buffer
   nread = 0;
   do {
      if (XGetWindowProperty((Display*)fDisplay, (Window) id, property,
                             nread/4, 1024, (Bool)del,
                             AnyPropertyType,
                             &actual_type, &actual_format,
                             &nitems, &bytes_after,
                             (unsigned char **) &data)
         != Success)
      break;

      if (actual_type != XA_STRING) break;

      text.Insert((Int_t) nread, (const char *) data, (Int_t) nitems);
      nread += nitems;
      XFree(data);

   } while (bytes_after > 0);

   nchar = (Int_t) nread;
}

//______________________________________________________________________________
void TGX11::TranslateCoordinates(Window_t src, Window_t dest, Int_t src_x,
                     Int_t src_y, Int_t &dest_x, Int_t &dest_y, Window_t &child)
{
   // TranslateCoordinates translates coordinates from the frame of
   // reference of one window to another. If the point is contained
   // in a mapped child of the destination, the id of that child is
   // returned as well.

   if (!src || !dest) return;

   Window xchild;

   XTranslateCoordinates((Display*)fDisplay, (Window) src, (Window) dest, src_x,
                         src_y, &dest_x, &dest_y, &xchild);
   child = (Window_t) xchild;
}

//______________________________________________________________________________
void TGX11::GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).

   if (!id) return;

   Window wdummy;
   UInt_t bdum, ddum;

   XGetGeometry((Display*)fDisplay, (Drawable) id, &wdummy, &x, &y, &w, &h, &bdum, &ddum);
}

//______________________________________________________________________________
void TGX11::FillPolygon(Window_t id, GContext_t gc, Point_t *points, Int_t npnt)
{
   // FillPolygon fills the region closed by the specified path.
   // The path is closed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.

   if (!id) return;

   XFillPolygon((Display*)fDisplay, (Window) id, (GC) gc, (XPoint *) points, npnt,
                Convex, CoordModeOrigin);
}

//______________________________________________________________________________
void TGX11::QueryPointer(Window_t id, Window_t &rootw, Window_t &childw,
                         Int_t &root_x, Int_t &root_y, Int_t &win_x,
                         Int_t &win_y, UInt_t &mask)
{
   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   // The pointer coordinates returned to win_x and win_y are relative to
   // the origin of the specified window. In this case, QueryPointer returns
   // the child that contains the pointer, if any, or else kNone to
   // childw. QueryPointer returns the current logical state of the
   // keyboard buttons and the modifier keys in mask.

   if (!id) return;

   Window xrootw, xchildw;
   UInt_t xmask;

   XQueryPointer((Display*)fDisplay, (Window) id, &xrootw, &xchildw,
                 &root_x, &root_y, &win_x, &win_y, &xmask);

   rootw  = (Window_t) xrootw;
   childw = (Window_t) xchildw;

   MapModifierState(mask, xmask, kFALSE);
}

//______________________________________________________________________________
void TGX11::SetForeground(GContext_t gc, ULong_t foreground)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).

   XSetForeground((Display*)fDisplay, (GC) gc, foreground);
}

//______________________________________________________________________________
void TGX11::SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.

   XSetClipRectangles((Display*)fDisplay, (GC) gc, x, y, (XRectangle *) recs, n, Unsorted);
}

//______________________________________________________________________________
void TGX11::Update(Int_t mode)
{
   // Flush (mode = 0, default) or synchronize (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.

   if (mode == 0)
      XFlush((Display*)fDisplay);
   if (mode == 1)
      XSync((Display*)fDisplay, False);
}

//______________________________________________________________________________
Region_t TGX11::CreateRegion()
{
   // Create a new empty region.

   return (Region_t) XCreateRegion();
}

//______________________________________________________________________________
void TGX11::DestroyRegion(Region_t reg)
{
   // Destroy region.

   XDestroyRegion((Region)reg);
}

//______________________________________________________________________________
void TGX11::UnionRectWithRegion(Rectangle_t *rect, Region_t src, Region_t dest)
{
   // Union of rectangle with a region.

   XRectangle *r = (XRectangle *) rect;   // 1 on 1 mapping
   XUnionRectWithRegion(r, (Region) src, (Region) dest);
}

//______________________________________________________________________________
Region_t TGX11::PolygonRegion(Point_t *points, Int_t np, Bool_t winding)
{
   // Create region for the polygon defined by the points array.
   // If winding is true use WindingRule else EvenOddRule as fill rule.

   XPoint *p = (XPoint *) points;
   return (Region_t) XPolygonRegion(p, np, winding ? WindingRule : EvenOddRule);
}

//______________________________________________________________________________
void TGX11::UnionRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the union of rega and regb and return result region.
   // The output region may be the same result region.

   XUnionRegion((Region) rega, (Region) regb, (Region) result);
}

//______________________________________________________________________________
void TGX11::IntersectRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the intersection of rega and regb and return result region.
   // The output region may be the same as the result region.

   XIntersectRegion((Region) rega, (Region) regb, (Region) result);
}

//______________________________________________________________________________
void TGX11::SubtractRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Subtract rega from regb.

   XSubtractRegion((Region) rega, (Region) regb, (Region) result);
}

//______________________________________________________________________________
void TGX11::XorRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Calculate the difference between the union and intersection of
   // two regions.

   XXorRegion((Region) rega, (Region) regb, (Region) result);
}

//______________________________________________________________________________
Bool_t TGX11::EmptyRegion(Region_t reg)
{
   // Return true if the region is empty.

   return (Bool_t) XEmptyRegion((Region) reg);
}

//______________________________________________________________________________
Bool_t TGX11::PointInRegion(Int_t x, Int_t y, Region_t reg)
{
   // Returns true if the point x,y is in the region.

   return (Bool_t) XPointInRegion((Region) reg, x, y);
}

//______________________________________________________________________________
Bool_t TGX11::EqualRegion(Region_t rega, Region_t regb)
{
   // Returns true if two regions are equal.

   return (Bool_t) XEqualRegion((Region) rega, (Region) regb);
}

//______________________________________________________________________________
void TGX11::GetRegionBox(Region_t reg, Rectangle_t *rect)
{
   // Return smallest enclosing rectangle.

   XClipBox((Region) reg, (XRectangle*) rect);
}

//______________________________________________________________________________
char **TGX11::ListFonts(const char *fontname, Int_t max, Int_t &count)
{
   // Return list of font names matching fontname regexp, like "-*-times-*".

   char **fontlist;
   Int_t fontcount = 0;
   fontlist = XListFonts((Display*)fDisplay, (char *)fontname, max, &fontcount);
   count = fontcount;
   return fontlist;
}

//______________________________________________________________________________
void TGX11::FreeFontNames(char **fontlist)
{
   // Free list of font names.

   XFreeFontNames(fontlist);
}

//______________________________________________________________________________
Drawable_t TGX11::CreateImage(UInt_t width, UInt_t height)
{
   // Create a client-side XImage. Returns handle to XImage.

   Int_t bitmap_pad;

   if (fDepth <= 8)
      bitmap_pad = 8;
   else if (fDepth <= 16)
      bitmap_pad = 16;
   else
      bitmap_pad = 32;

   XImage *xim = XCreateImage((Display*)fDisplay, fVisual, fDepth, ZPixmap,
                              0, 0, width, height, bitmap_pad, 0);

   // use calloc since Xlib will use free() in XDestroyImage
   if (xim) xim->data = (char *) calloc(xim->bytes_per_line * xim->height, 1);

   return (Drawable_t) xim;
}

//______________________________________________________________________________
void TGX11::GetImageSize(Drawable_t img, UInt_t &width, UInt_t &height)
{
   // Get size of XImage img.

   width  = ((XImage*)img)->width;
   height = ((XImage*)img)->height;
}

//______________________________________________________________________________
void TGX11::PutPixel(Drawable_t img, Int_t x, Int_t y, ULong_t pixel)
{
   // Set pixel at specified location in XImage img.

   XPutPixel((XImage*) img, x, y, pixel);
}

//______________________________________________________________________________
void TGX11::PutImage(Drawable_t win, GContext_t gc, Drawable_t img, Int_t dx,
                     Int_t dy, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Put (x,y,w,h) part of image img in window win at position dx,dy.

   if (!win) return;

   XPutImage((Display*)fDisplay, (Drawable) win, (GC) gc, (XImage*) img,
             x, y, dx, dy, w, h);
}

//______________________________________________________________________________
void TGX11::DeleteImage(Drawable_t img)
{
   // Destroy XImage img.

   XDestroyImage((XImage*) img);
}

//______________________________________________________________________________
void TGX11::ShapeCombineMask(Window_t id, Int_t x, Int_t y, Pixmap_t mask)
{
   // The Nonrectangular Window Shape Extension adds nonrectangular
   // windows to the System.
   // This allows for making shaped (partially transparent) windows

   XShapeCombineMask((Display*)fDisplay, (Window) id, ShapeBounding, x, y,
                     (Pixmap) mask, ShapeSet);
}

//______________________________________________________________________________
UInt_t TGX11::ScreenWidthMM() const
{
   // Returns the width of the screen in millimeters.

   return (UInt_t)WidthMMOfScreen(DefaultScreenOfDisplay((Display*)fDisplay));
}

//______________________________________________________________________________
void TGX11::DeleteProperty(Window_t win, Atom_t& prop)
{
   // Deletes the specified property only if the property was defined on the
   // specified window and causes the X server to generate a PropertyNotify
   // event on the window unless the property does not exist.

   XDeleteProperty((Display*)fDisplay, win, prop);
}

//______________________________________________________________________________
Int_t TGX11::GetProperty(Window_t win, Atom_t prop, Long_t offset, Long_t length,
                         Bool_t del, Atom_t req_type, Atom_t *act_type,
                         Int_t *act_format, ULong_t *nitems, ULong_t *bytes,
                         unsigned char **prop_list)
{
   // Returns the actual type of the property; the actual format of the property;
   // the number of 8-bit, 16-bit, or 32-bit items transferred; the number of
   // bytes remaining to be read in the property; and a pointer to the data
   // actually returned.

   return XGetWindowProperty((Display*)fDisplay, win, prop, offset, length, del, req_type,
                             act_type, act_format, nitems, bytes, prop_list);
}

//______________________________________________________________________________
void TGX11::ChangeActivePointerGrab(Window_t /*win*/, UInt_t mask, Cursor_t cur)
{
   // Changes the specified dynamic parameters if the pointer is actively
   // grabbed by the client.

   UInt_t xevmask;
   MapEventMask(mask, xevmask);
   if (cur == kNone)
      XChangeActivePointerGrab((Display*)fDisplay, xevmask, fCursors[kHand], CurrentTime);
   else
      XChangeActivePointerGrab((Display*)fDisplay, xevmask, cur, CurrentTime);
}

//______________________________________________________________________________
void TGX11::ConvertSelection(Window_t win, Atom_t &sel, Atom_t &target,
                             Atom_t &prop, Time_t &stamp)
{
   // Requests that the specified selection be converted to the specified
   // target type.

   XConvertSelection((Display*)fDisplay, sel, target, prop, win, stamp);
}

//______________________________________________________________________________
Bool_t TGX11::SetSelectionOwner(Window_t owner, Atom_t &sel)
{
   // Changes the owner and last-change time for the specified selection

   return XSetSelectionOwner((Display*)fDisplay, sel, owner, CurrentTime);
}

//______________________________________________________________________________
void TGX11::ChangeProperties(Window_t id, Atom_t property, Atom_t type,
                             Int_t format, UChar_t *data, Int_t len)
{
   // This function alters the property for the specified window and
   // causes the X server to generate a PropertyNotify event on that
   // window.

   if (!id) return;

   XChangeProperty((Display*)fDisplay, (Window) id, (Atom) property, (Atom) type,
                   format, PropModeReplace, data, len);
}

//______________________________________________________________________________
void TGX11::SetDNDAware(Window_t win, Atom_t *typelist)
{
   // Add XdndAware property and the list of drag and drop types to the
   // Window win.

   unsigned char version = 4;
   Atom_t dndaware = InternAtom("XdndAware", kFALSE);
   XChangeProperty((Display*)fDisplay, (Window) win, (Atom) dndaware, (Atom) XA_ATOM,
                   32, PropModeReplace, (unsigned char *) &version, 1);

   if (typelist) {
      int n;

      for (n = 0; typelist[n]; n++) { }
      if (n > 0) {
         XChangeProperty((Display*)fDisplay, win, dndaware, XA_ATOM, 32, PropModeAppend,
                         (unsigned char *) typelist, n);
      }
   }
}

//______________________________________________________________________________
void TGX11::SetTypeList(Window_t win, Atom_t prop, Atom_t *typelist)
{
   // Add the list of drag and drop types to the Window win.

   if (typelist) {
      int n;
      for (n = 0; typelist[n]; n++) { }
      if (n > 0) {
         XChangeProperty((Display*)fDisplay, win, prop, XA_ATOM, 32, PropModeAppend,
                         (unsigned char *) typelist, n);
      }
   }
}

//______________________________________________________________________________
Window_t TGX11::FindRWindow(Window_t win, Window_t dragwin, Window_t input,
                            int x, int y, int maxd)
{
   // Recursively search in the children of Window for a Window which is at
   // location x, y and is DND aware, with a maximum depth of maxd.
   // Possibility to exclude dragwin and input.

   WindowAttributes_t wattr;
   static Atom_t *dndTypeList = 0;

   if (dndTypeList == 0) {
      dndTypeList = new Atom_t[3];
      dndTypeList[0] = InternAtom("application/root", kFALSE);
      dndTypeList[1] = InternAtom("text/uri-list", kFALSE);
      dndTypeList[2] = 0;
   }

   if (maxd <= 0) return kNone;

   if (win == dragwin || win == input) return kNone;

   GetWindowAttributes(win, wattr);
   if (wattr.fMapState != kIsUnmapped &&
       x >= wattr.fX && x < wattr.fX + wattr.fWidth &&
       y >= wattr.fY && y < wattr.fY + wattr.fHeight) {

      if (IsDNDAware(win, dndTypeList)) return win;

      Window r, p, *children;
      UInt_t numch;
      int i;

      if (XQueryTree((Display*)fDisplay, win, &r, &p, &children, &numch)) {
         if (children && numch > 0) {
            r = kNone;
            // upon return from XQueryTree, children are listed in the current
            // stacking order, from bottom-most (first) to top-most (last)
            for (i = numch-1; i >= 0; --i) {
               r = FindRWindow((Window_t)children[i], dragwin, input,
                               x - wattr.fX, y - wattr.fY, maxd-1);
               if (r != kNone) break;
            }
            XFree(children);
            if (r != kNone) return r;
         }
         return kNone; //win;   // ?!?
      }
   }
   return kNone;
}

//______________________________________________________________________________
Bool_t TGX11::IsDNDAware(Window_t win, Atom_t *typelist)
{
   // Checks if Window win is DND aware, and knows any of the DND formats
   // passed in argument.

   Atom_t  actual;
   Int_t   format;
   ULong_t count, remaining;
   unsigned char *data = 0;
   Atom_t *types, *t;
   Int_t   result = kTRUE;
   static Atom_t dndaware = kNone;

   if (win == kNone) return kFALSE;

   if (dndaware == kNone)
      dndaware = InternAtom("XdndAware", kFALSE);

   XGetWindowProperty((Display*)fDisplay, win, dndaware, 0, 0x8000000L, kFALSE,
                      XA_ATOM, &actual, &format, &count, &remaining, &data);

   if ((actual != XA_ATOM) || (format != 32) || (count == 0) || !data) {
      if (data) XFree(data);
      return kFALSE;
   }

   types = (Atom_t *) data;

   if ((count > 1) && typelist) {
      result = kFALSE;
      for (t = typelist; *t; t++) {
         for (ULong_t j = 1; j < count; j++) {
            if (types[j] == *t) {
               result = kTRUE;
               break;
            }
         }
         if (result) break;
      }
   }
   XFree(data);
   return result;
}
