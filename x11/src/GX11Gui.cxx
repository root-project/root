// @(#)root/x11:$Name:  $:$Id: GX11Gui.cxx,v 1.4 2000/07/06 16:49:39 rdm Exp $
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
// an implementation of the abstract TVirtualX class. The companion class    //
// for Win32 is TGWin32.                                                //
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
#include <unistd.h>

#include "TGX11.h"
#include "TROOT.h"
#include "TError.h"
#include "TException.h"
#include "TClassTable.h"
#include "KeySymbols.h"


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
   { XK_KP_Equal,        kKey_Equal },
   { XK_KP_Multiply,     kKey_Asterisk },
   { XK_KP_Add,          kKey_Plus },
   { XK_KP_Separator,    kKey_Comma },
   { XK_KP_Subtract,     kKey_Minus },
   { XK_KP_Decimal,      kKey_Period },
   { XK_KP_Divide,       kKey_Slash },
   { 0,                  (EKeySym) 0 }
};



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
   ::Error("RootX11ErrorHandler", "%s (XID: %u)", msg, err->resourceid);
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

   XMapWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MapSubwindows(Window_t id)
{
   // Map sub windows.

   XMapSubwindows(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.

   XMapRaised(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::UnmapWindow(Window_t id)
{
   // Unmap window from screen.

   XUnmapWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::DestroyWindow(Window_t id)
{
   // Destroy window.

   XDestroyWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.

   XRaiseWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.

   XLowerWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
void TGX11::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.

   XMoveWindow(fDisplay, (Window) id, x, y);
}

//______________________________________________________________________________
void TGX11::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Move and resize a window.

   XMoveResizeWindow(fDisplay, (Window) id, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.

   XResizeWindow(fDisplay, (Window) id, w, h);
}

//______________________________________________________________________________
void TGX11::IconifyWindow(Window_t id)
{
   // Iconify the window.

   XIconifyWindow(fDisplay, (Window) id, DefaultScreen(fDisplay));
}

//______________________________________________________________________________
void TGX11::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.

   XSetWindowBackground(fDisplay, (Window) id, color);
}

//______________________________________________________________________________
void TGX11::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.

   XSetWindowBackgroundPixmap(fDisplay, (Window) id, (Pixmap) pxm);
}

//______________________________________________________________________________
Window_t TGX11::CreateWindow(Window_t parent, Int_t x, Int_t y,
                             UInt_t w, UInt_t h, UInt_t border,
                             Int_t depth, UInt_t clss,
                             void *visual, SetWindowAttributes_t *attr)
{
   // Return handle to newly created X window.

   XSetWindowAttributes xattr;
   ULong_t              xmask = 0;

   if (attr)
      MapSetWindowAttributes(attr, xmask, xattr);

   return (Window_t) XCreateWindow(fDisplay, (Window) parent, x, y,
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
   }
}

//______________________________________________________________________________
void TGX11::MapSetWindowAttributes(SetWindowAttributes_t *attr,
                                   ULong_t &xmask, XSetWindowAttributes &xattr)
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
                        ULong_t &xmask, XGCValues &xgval, Bool_t tox)
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
         xmask |= GCFont;
         xgval.font = (Font) gval.fFont;
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
         xgval.dashes = gval.fDashes;
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
         gval.fDashes = xgval.dashes;
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

   XWindowAttributes xattr;

   XGetWindowAttributes(fDisplay, id, &xattr);

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
   attr.fColormap           = (Colormap_t) xattr.colormap;
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
      // very first call before any X-call !!
      if (!XInitThreads())
         Warning("OpenDisplay", "system has no X11 thread support");
#endif

   Display *dpy;
   if (!(dpy = XOpenDisplay(dpyName)))
      return -1;

   // Set custom X11 error handlers
   XSetErrorHandler(RootX11ErrorHandler);
   XSetIOErrorHandler(RootX11IOErrorHandler);

   if (gDebug > 4)
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

   XCloseDisplay(fDisplay);
}

//______________________________________________________________________________
Display_t TGX11::GetDisplay()
{
   // Returns handle to display (might be usefull in some cases where
   // direct X11 manipulation outside of TVirtualX is needed, e.g. GL interface).

   return (Display_t) fDisplay;
}

//______________________________________________________________________________
Atom_t TGX11::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   Atom a = XInternAtom(fDisplay, (char *)atom_name, (Bool)only_if_exist);

   if (a == None) return kNone;
   return (Atom_t) a;
}

//______________________________________________________________________________
Window_t TGX11::GetDefaultRootWindow()
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().

   return (Window_t) XDefaultRootWindow(fDisplay);
}

//______________________________________________________________________________
Window_t TGX11::GetParent(Window_t id)
{
   // Return the parent of the window.

   Window  root, parent;
   Window *children = 0;
   UInt_t  nchildren;

   XQueryTree(fDisplay, (Window) id, &root, &parent, &children, &nchildren);

   if (children) XFree(children);

   return (Window_t) parent;
}


//______________________________________________________________________________
FontStruct_t TGX11::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise an opaque pointer to the FontStruct_t.
   // Free the loaded font using DeleteFont().

   XFontStruct *fs = XLoadQueryFont(fDisplay, (char *)font_name);
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

   XFreeFont(fDisplay, (XFontStruct *) fs);
}

//______________________________________________________________________________
GContext_t TGX11::CreateGC(Drawable_t id, GCValues_t *gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).

   XGCValues xgval;
   ULong_t   xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, xgval);

   GC gc = XCreateGC(fDisplay, (Drawable) id, xmask, &xgval);

   return (GContext_t) gc;
}

//______________________________________________________________________________
void TGX11::ChangeGC(GContext_t gc, GCValues_t *gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.

   XGCValues xgval;
   ULong_t   xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, xgval);

   XChangeGC(fDisplay, (GC) gc, xmask, &xgval);
}

//______________________________________________________________________________
void TGX11::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. Both org and dest must exist.

   GCValues_t gval;
   XGCValues  xgval;
   ULong_t    xmask;

   if (!mask) {
      // in this case copy all fields
      mask = (Mask_t)-1;
   }

   gval.fMask = mask;  // only set fMask used to convert to xmask
   MapGCValues(gval, xmask, xgval);

   XCopyGC(fDisplay, (GC) org, xmask, (GC) dest);
}

//______________________________________________________________________________
void TGX11::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.

   XFreeGC(fDisplay, (GC) gc);
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

   XDefineCursor(fDisplay, (Window) id, (Cursor) curid);
}

//______________________________________________________________________________
Pixmap_t TGX11::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.

   return (Pixmap_t) XCreatePixmap(fDisplay, (Drawable) id, w, h,
                        DefaultDepth(fDisplay, DefaultScreen(fDisplay)));
}

//______________________________________________________________________________
Pixmap_t TGX11::CreatePixmap(Drawable_t id, const char *bitmap,
            UInt_t width, UInt_t height, ULong_t forecolor, ULong_t backcolor,
            Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.

   return (Pixmap_t) XCreatePixmapFromBitmapData(fDisplay, id, (char *)bitmap,
                           width, height, forecolor, backcolor, depth);
}

//______________________________________________________________________________
Pixmap_t TGX11::CreateBitmap(Drawable_t id, const char *bitmap,
                             UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

   return (Pixmap_t) XCreateBitmapFromData(fDisplay, id, (char *)bitmap,
                                           width, height);
}

//______________________________________________________________________________
void TGX11::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.

   XFreePixmap(fDisplay, (Pixmap) pmap);
}

//______________________________________________________________________________
void TGX11::MapPictureAttributes(PictureAttributes_t &attr, XpmAttributes &xpmattr,
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
      }
      if ((xmask & XpmDepth)) {
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

#ifdef XpmVersion
   XpmAttributes xpmattr;

   MapPictureAttributes(attr, xpmattr);

   Int_t res = XpmReadFileToPixmap(fDisplay, id, (char*)filename,
                                   (Pixmap*)&pict, (Pixmap*)&pict_mask, &xpmattr);

   MapPictureAttributes(attr, xpmattr, kFALSE);
   XpmFreeAttributes(&xpmattr);

   if (res == XpmSuccess || res == XpmColorError)
      return kTRUE;

   if (pict) {
      XFreePixmap(fDisplay, (Pixmap)pict);
      pict = kNone;
   }
   if (pict_mask) {
      XFreePixmap(fDisplay, (Pixmap)pict_mask);
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
   XpmAttributes xpmattr;

   MapPictureAttributes(attr, xpmattr);

   Int_t res = XpmCreatePixmapFromData(fDisplay, id, data, (Pixmap*)&pict,
                                       (Pixmap*)&pict_mask, &xpmattr);

   MapPictureAttributes(attr, xpmattr, kFALSE);
   XpmFreeAttributes(&xpmattr);

   if (res == XpmSuccess || res == XpmColorError)
      return kTRUE;

   if (pict) {
      XFreePixmap(fDisplay, (Pixmap)pict);
      pict = kNone;
   }
   if (pict_mask) {
      XFreePixmap(fDisplay, (Pixmap)pict_mask);
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

   XSetDashes(fDisplay, (GC) gc, offset, (char *)dash_list, n);
}

//______________________________________________________________________________
void TGX11::MapColorStruct(ColorStruct_t *color, XColor &xcolor)
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

   if (XParseColor(fDisplay, (Colormap)cmap, (char *)cname, &xc)) {
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

   XColor xc;

   MapColorStruct(&color, xc);

   int status = XAllocColor(fDisplay, (Colormap)cmap, &xc);
   color.fPixel = xc.pixel;

   return status != 0 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGX11::QueryColor(Colormap_t cmap, ColorStruct_t &color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   XColor xc;

   xc.pixel = color.fPixel;

   XQueryColor(fDisplay, (Colormap)cmap, &xc);

   color.fRed   = xc.red;
   color.fGreen = xc.green;
   color.fBlue  = xc.blue;
}

//______________________________________________________________________________
Int_t TGX11::EventsPending()
{
   // Returns number of pending events.

   return XPending(fDisplay);
}

//______________________________________________________________________________
void TGX11::NextEvent(Event_t &event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   XEvent xev;

   XNextEvent(fDisplay, &xev);

   // fill in Event_t
   MapEvent(event, xev, kFALSE);
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
void TGX11::MapEvent(Event_t &ev, XEvent &xev, Bool_t tox)
{
   // Map Event_t structure to XEvent structure. If tox is false
   // map the other way.

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

      xev.xany.window     = (Window) ev.fWindow;
      xev.xany.send_event = (Bool) ev.fSendEvent;
      xev.xany.display    = fDisplay;

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
   }
}

//______________________________________________________________________________
void TGX11::Bell(Int_t percent)
{
   // Sound bell. Percent is loudness from -100% .. 100%.

   XBell(fDisplay, percent);
}

//______________________________________________________________________________
void TGX11::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                     Int_t src_x, Int_t src_y, UInt_t width, UInt_t height,
                     Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.

   XCopyArea(fDisplay, src, dest, (GC) gc, src_x, src_y, width, height,
             dest_x, dest_y);
}

//______________________________________________________________________________
void TGX11::ChangeWindowAttributes(Window_t id, SetWindowAttributes_t *attr)
{
   // Change window attributes.

   XSetWindowAttributes xattr;
   ULong_t              xmask = 0;

   if (attr)
      MapSetWindowAttributes(attr, xmask, xattr);

   XChangeWindowAttributes(fDisplay, (Window) id, xmask, &xattr);

   if (attr && (attr->fMask & kWABorderWidth))
      XSetWindowBorderWidth(fDisplay, (Window) id, attr->fBorderWidth);
}

//______________________________________________________________________________
void TGX11::ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                           UChar_t *data, Int_t len)
{
   // This function alters the property for the specified window and
   // causes the X server to generate a PropertyNotify event on that
   // window.

   XChangeProperty(fDisplay, (Window) id, (Atom) property, (Atom) type,
                   8, PropModeReplace, data, len);
}

//______________________________________________________________________________
void TGX11::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1, Int_t x2, Int_t y2)
{
   // Draw a line.

   XDrawLine(fDisplay, (Drawable) id, (GC) gc, x1, y1, x2, y2);
}

//______________________________________________________________________________
void TGX11::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.

   XClearArea(fDisplay, (Window) id, x, y, w, h, False);
}

//______________________________________________________________________________
Bool_t TGX11::CheckEvent(Window_t id, EGEventType type, Event_t &ev)
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.

   Event_t tev;
   XEvent  xev;

   tev.fType = type;
   MapEvent(tev, xev);

   Bool r = XCheckTypedWindowEvent(fDisplay, (Window) id, xev.type, &xev);

   if (r)
      MapEvent(ev, xev, kFALSE);

   return r ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGX11::SendEvent(Window_t id, Event_t *ev)
{
   // Send event ev to window id.

   if (!ev) return;

   XEvent xev;

   MapEvent(*ev, xev);

   XSendEvent(fDisplay, (Window) id, False, None, &xev);
}

//______________________________________________________________________________
void TGX11::WMDeleteNotify(Window_t id)
{
   // Tell WM to send message when window is closed via WM.

   XSetWMProtocols(fDisplay, (Window) id, &gWM_DELETE_WINDOW, 1);
}

//______________________________________________________________________________
void TGX11::SetKeyAutoRepeat(Bool_t on)
{
   // Turn key auto repeat on or off.

   if (on)
      XAutoRepeatOn(fDisplay);
   else
      XAutoRepeatOff(fDisplay);
}

//______________________________________________________________________________
void TGX11::GrabKey(Window_t id, Int_t keycode, UInt_t modifier, Bool_t grab)
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.

   UInt_t xmod;

   MapModifierState(modifier, xmod);

   if (grab)
      XGrabKey(fDisplay, keycode, xmod, (Window) id, True,
               GrabModeAsync, GrabModeAsync);
   else
      XUngrabKey(fDisplay, keycode, xmod, (Window) id);
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

   UInt_t xmod;

   MapModifierState(modifier, xmod);

   if (grab) {
      UInt_t xevmask;
      MapEventMask(evmask, xevmask);

      XGrabButton(fDisplay, button, xmod, (Window) id, True, xevmask,
                  GrabModeAsync, GrabModeAsync, (Window) confine,
                  (Cursor) cursor);
   } else
      XUngrabButton(fDisplay, button, xmod, (Window) id);
}

//______________________________________________________________________________
void TGX11::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                        Cursor_t cursor, Bool_t grab)
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.

   if (grab) {
      UInt_t xevmask;
      MapEventMask(evmask, xevmask);

      XGrabPointer(fDisplay, (Window) id, True,
                   xevmask, GrabModeAsync, GrabModeAsync, (Window) confine,
                   (Cursor) cursor, CurrentTime);
   } else
      XUngrabPointer(fDisplay, CurrentTime);
}

//______________________________________________________________________________
void TGX11::SetWindowName(Window_t id, char *name)
{
   // Set window name.

   XTextProperty wname;

   if (XStringListToTextProperty(&name, 1, &wname) == 0) {
      Error("SetWindowName", "cannot allocate window name \"%s\"", name);
      return;
   }
   XSetWMName(fDisplay, (Window) id, &wname);
   XFree(wname.value);
}

//______________________________________________________________________________
void TGX11::SetIconName(Window_t id, char *name)
{
   // Set window icon name.

   XTextProperty wname;

   if (XStringListToTextProperty(&name, 1, &wname) == 0) {
      Error("SetIconName", "cannot allocate icon name \"%s\"", name);
      return;
   }
   XSetWMIconName(fDisplay, (Window) id, &wname);
   XFree(wname.value);
}

//______________________________________________________________________________
void TGX11::SetIconPixmap(Window_t id, Pixmap_t pic)
{
   // Set pixmap the WM can use when the window is iconized.

   XWMHints hints;

   hints.flags = IconPixmapHint;
   hints.icon_pixmap = (Pixmap) pic;

   XSetWMHints(fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetClassHints(Window_t id, char *className, char *resourceName)
{
   // Set the windows class and resource name.

   XClassHint class_hints;

   class_hints.res_class = className;
   class_hints.res_name  = resourceName;
   XSetClassHint(fDisplay, (Window) id, &class_hints);
}

//______________________________________________________________________________
void TGX11::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs, UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).

   MWMHintsProperty_t prop;

   prop.fDecorations = value;
   prop.fFunctions   = funcs;
   prop.fInputMode   = input;
   prop.fFlags       = kMWMHintsDecorations |
                       kMWMHintsFunctions   |
                       kMWMHintsInputMode;

   XChangeProperty(fDisplay, (Window) id, gMOTIF_WM_HINTS, gMOTIF_WM_HINTS, 32,
                   PropModeReplace, (UChar_t *)&prop, kPropMWMHintElements);
}

//______________________________________________________________________________
void TGX11::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   // Tell the window manager the desired window position.

   XSizeHints hints;

   hints.flags = USPosition | PPosition;
   hints.x = x;
   hints.y = y;

   XSetWMNormalHints(fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   // Tell the window manager the desired window size.

   XSizeHints hints;

   hints.flags = USSize | PSize | PBaseSize;
   hints.width = hints.base_width = w;
   hints.height = hints.base_height = h;

   XSetWMNormalHints(fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                           UInt_t wmax, UInt_t hmax,
                           UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   XSizeHints hints;

   hints.flags = PMinSize | PMaxSize | PResizeInc;
   hints.min_width   = (Int_t)wmin;
   hints.max_width   = (Int_t)wmax;
   hints.min_height  = (Int_t)hmin;
   hints.max_height  = (Int_t)hmax;
   hints.width_inc   = (Int_t)winc;
   hints.height_inc  = (Int_t)hinc;

   XSetWMNormalHints(fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMState(Window_t id, EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.

   XWMHints hints;
   Int_t    xstate = NormalState;

   if (state == kNormalState)
      xstate = NormalState;
   if (state == kIconicState)
      xstate = IconicState;

   hints.flags = StateHint;
   hints.initial_state = xstate;

   XSetWMHints(fDisplay, (Window) id, &hints);
}

//______________________________________________________________________________
void TGX11::SetWMTransientHint(Window_t id, Window_t main_id)
{
   // Tell window manager that window is a transient window of main.

   XSetTransientForHint(fDisplay, (Window) id, (Window) main_id);
}

//______________________________________________________________________________
void TGX11::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                       const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y).

   XDrawString(fDisplay, (Drawable) id, (GC) gc, x, y, (char *) s, len);
}

//______________________________________________________________________________
Int_t TGX11::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font.

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
   // context to get is encoded in the GCValues::fMask member.

   XGCValues xgval;
   ULong_t   xmask;

   MapGCValues(gval, xmask, xgval);

   XGetGCValues(fDisplay, (GC) gc, xmask, &xgval);

   MapGCValues(gval, xmask, xgval, kFALSE);
}

//______________________________________________________________________________
FontStruct_t TGX11::GetFontStruct(FontH_t fh)
{
   // Retrieve associated font structure once we have the font handle.
   // Free returned FontStruct_t using FreeFontStruct().

   XFontStruct *fs;

   fs = XQueryFont(fDisplay, (Font) fh);

   return (FontStruct_t) fs;
}

//______________________________________________________________________________
void TGX11::FreeFontStruct(FontStruct_t fs)
{
   // Free font structure returned by GetFontStruct().

   // in XFree86 4.0 XFreeFontInfo() is broken, ok again in 4.0.1
   static int xfree86_400 = -1;
   if (xfree86_400 == -1) {
      if (strstr(XServerVendor(fDisplay), "XFree86") &&
          XVendorRelease(fDisplay) == 4000)
         xfree86_400 = 1;
      else
         xfree86_400 = 0;
      //printf("Vendor: %s, Release = %d\n", XServerVendor(fDisplay), XVendorRelease(fDisplay));
   }

   if (xfree86_400 == 0)
      XFreeFontInfo(0, (XFontStruct *) fs, 1);
}

//______________________________________________________________________________
void TGX11::ClearWindow(Window_t id)
{
   // Clear window.

   XClearWindow(fDisplay, (Window) id);
}

//______________________________________________________________________________
Int_t TGX11::KeysymToKeycode(UInt_t keysym)
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).

   UInt_t xkeysym;
   MapKeySym(keysym, xkeysym);

   return XKeysymToKeycode(fDisplay, xkeysym);
}

//______________________________________________________________________________
void TGX11::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a filled rectangle. Filling is done according to the gc.

   XFillRectangle(fDisplay, (Drawable) id, (GC) gc, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw a rectangle outline.

   XDrawRectangle(fDisplay, (Drawable) id, (GC) gc, x, y, w, h);
}

//______________________________________________________________________________
void TGX11::DrawSegments(Drawable_t id, GContext_t gc, Segment_t *seg, Int_t nseg)
{
   // Draws multiple line segments. Each line is specified by a pair of points.

   XDrawSegments(fDisplay, (Drawable) id, (GC) gc, (XSegment *) seg, nseg);
}

//______________________________________________________________________________
void TGX11::SelectInput(Window_t id, UInt_t evmask)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   UInt_t xevmask;

   MapEventMask(evmask, xevmask);

   XSelectInput(fDisplay, (Window) id, xevmask);
}

//______________________________________________________________________________
void TGX11::SetInputFocus(Window_t id)
{
   // Set keyboard input focus to window id.

   XSetInputFocus(fDisplay, (Window) id, RevertToParent, CurrentTime);
}

//______________________________________________________________________________
Window_t TGX11::GetPrimarySelectionOwner()
{
   // Returns the window id of the current owner of the primary selection.
   // That is the window in which, for example some text is selected.

   return (Window_t) XGetSelectionOwner(fDisplay, XA_PRIMARY);
}

//______________________________________________________________________________
void TGX11::SetPrimarySelectionOwner(Window_t id)
{
   // Makes the window id the current owner of the primary selection.
   // That is the window in which, for example some text is selected.

   XSetSelectionOwner(fDisplay, XA_PRIMARY, id, CurrentTime);
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

   XConvertSelection(fDisplay, XA_PRIMARY, XA_STRING, (Atom) clipboard,
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

   MapEvent(*event, xev);

   int n = XLookupString(&xev.xkey, buf, buflen-1, &xkeysym, 0);
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

   Atom actual_type, property = (Atom) atom;
   int  actual_format;
   unsigned long nitems, bytes_after, nread;
   unsigned char *data;

   nchar = 0;
   text  = "";

   if (property == None) return;

   // get past buffer
   nread = 0;
   do {
      if (XGetWindowProperty(fDisplay, (Window) id, property,
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

   Window xchild;

   XTranslateCoordinates(fDisplay, (Window) src, (Window) dest, src_x,
                         src_y, &dest_x, &dest_y, &xchild);
   child = (Window_t) xchild;
}

//______________________________________________________________________________
void TGX11::GetWindowSize(Drawable_t id, Int_t &x, Int_t &y, UInt_t &w, UInt_t &h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).

   Window wdummy;
   UInt_t bdum, ddum;

   XGetGeometry(fDisplay, (Drawable) id, &wdummy, &x, &y, &w, &h, &bdum, &ddum);
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

   XFillPolygon(fDisplay, (Window) id, (GC) gc, (XPoint *) points, npnt,
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

   Window xrootw, xchildw;
   UInt_t xmask;

   XQueryPointer(fDisplay, (Window) id, &xrootw, &xchildw,
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

   XSetForeground(fDisplay, (GC) gc, foreground);
}

//______________________________________________________________________________
void TGX11::SetClipRectangles(GContext_t gc, Int_t x, Int_t y, Rectangle_t *recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.

   XSetClipRectangles(fDisplay, (GC) gc, x, y, (XRectangle *) recs, n, Unsorted);
}

//______________________________________________________________________________
void TGX11::Update(Int_t mode)
{
   // Flush (mode = 0, default) or synchronize (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.

   if (mode == 0)
      XFlush(fDisplay);
   if (mode == 1)
      XSync(fDisplay, False);
}
