// @(#)root/win32gdk:$Name:  $:$Id: GWin32Gui.cxx,v 1.2 2001/11/30 12:39:20 rdm Exp $
// Author: Bertrand Bellenot, Fons Rademakers   27/11/01

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32 (GUI related part)                                           //
//                                                                      //
// This class is the basic interface to the X11 graphics system. It is  //
// an implementation of the abstract TVirtualX class.                   //
//                                                                      //
// This file contains the implementation of the GUI methods of the      //
// TGWin32 class. Most of the methods are used by the machine           //
// independent GUI classes (libGUI.so).                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "gdk/gdkkeysyms.h"
#include "xatom.h"

#include "TGFrame.h"
#include "TGWin32.h"
#include "TROOT.h"
#include "TError.h"
#include "TException.h"
#include "TClassTable.h"
#include "KeySymbols.h"

int gdk_debug_level;

extern GdkAtom clipboard_atom;

/* Key masks. Used as modifiers to GrabButton and GrabKey, results of QueryPointer,
   state in various key-, mouse-, and button-related events. */

//---- MWM Hints stuff

struct MWMHintsProperty_t {
   Handle_t fFlags;
   Handle_t fFunctions;
   Handle_t fDecorations;
   Int_t fInputMode;
};

//---- hints

const ULong_t kMWMHintsFunctions = BIT(0);
const ULong_t kMWMHintsDecorations = BIT(1);
const ULong_t kMWMHintsInputMode = BIT(2);

const Int_t kPropMotifWMHintsElements = 4;
const Int_t kPropMWMHintElements = kPropMotifWMHintsElements;


//---- Key symbol mapping

struct KeySymbolMap_t {
   KeySym fXKeySym;
   EKeySym fKeySym;
};

static char *keyCodeToString[] = {
   "",                          /* 0x000 */
   "",                          /* 0x001, VK_LBUTTON */
   "",                          /* 0x002, VK_RBUTTON */
   "",                          /* 0x003, VK_CANCEL */
   "",                          /* 0x004, VK_MBUTTON */
   "",                          /* 0x005 */
   "",                          /* 0x006 */
   "",                          /* 0x007 */
   "\015",                      /* 0x008, VK_BACK */
   "\t",                        /* 0x009, VK_TAB */
   "",                          /* 0x00A */
   "",                          /* 0x00B */
   "",                          /* 0x00C, VK_CLEAR */
   "\r",                        /* 0x00D, VK_RETURN */
   "",                          /* 0x00E */
   "",                          /* 0x00F */
   "",                          /* 0x010, VK_SHIFT */
   "",                          /* 0x011, VK_CONTROL */
   "",                          /* 0x012, VK_MENU */
   "",                          /* 0x013, VK_PAUSE */
   "",                          /* 0x014, VK_CAPITAL */
   "",                          /* 0x015, VK_KANA */
   "",                          /* 0x016 */
   "",                          /* 0x017 */
   "",                          /* 0x018 */
   "",                          /* 0x019, VK_KANJI */
   "",                          /* 0x01A */
   "",                          /* 0x01B, VK_ESCAPE */
   "",                          /* 0x01C, VK_CONVERT */
   "",                          /* 0x01D, VK_NONCONVERT */
   "",                          /* 0x01E */
   "",                          /* 0x01F */
   " ",                         /* 0x020, VK_SPACE */
   "",                          /* 0x021, VK_PRIOR */
   "",                          /* 0x022, VK_NEXT */
   "",                          /* 0x023, VK_END */
   "",                          /* 0x024, VK_HOME */
   "",                          /* 0x025, VK_LEFT */
   "",                          /* 0x026, VK_UP */
   "",                          /* 0x027, VK_RIGHT */
   "",                          /* 0x028, VK_DOWN */
   "",                          /* 0x029, VK_SELECT */
   "",                          /* 0x02A, VK_PRINT */
   "",                          /* 0x02B, VK_EXECUTE */
   "",                          /* 0x02C, VK_SNAPSHOT */
   "",                          /* 0x02D, VK_INSERT */
   "\037",                      /* 0x02E, VK_DELETE */
   "",                          /* 0x02F, VK_HELP */
};

//---- Mapping table of all non-trivial mappings (the ASCII keys map
//---- one to one so are not included)

static KeySymbolMap_t gKeyMap[] = {
   {GDK_Escape, kKey_Escape},
   {GDK_Tab, kKey_Tab},
#ifndef GDK_ISO_Left_Tab
   {0xFE20, kKey_Backtab},
#else
   {GDK_ISO_Left_Tab, kKey_Backtab},
#endif
   {GDK_BackSpace, kKey_Backspace},
   {GDK_Return, kKey_Return},
   {GDK_Insert, kKey_Insert},
   {GDK_Delete, kKey_Delete},
   {GDK_Clear, kKey_Delete},
   {GDK_Pause, kKey_Pause},
   {GDK_Print, kKey_Print},
   {0x1005FF60, kKey_SysReq},   // hardcoded Sun SysReq
   {0x1007ff00, kKey_SysReq},   // hardcoded X386 SysReq
   {GDK_Home, kKey_Home},       // cursor movement
   {GDK_End, kKey_End},
   {GDK_Left, kKey_Left},
   {GDK_Up, kKey_Up},
   {GDK_Right, kKey_Right},
   {GDK_Down, kKey_Down},
   {GDK_Prior, kKey_Prior},
   {GDK_Next, kKey_Next},
   {GDK_Shift_L, kKey_Shift},   // modifiers
   {GDK_Shift_R, kKey_Shift},
   {GDK_Shift_Lock, kKey_Shift},
   {GDK_Control_L, kKey_Control},
   {GDK_Control_R, kKey_Control},
   {GDK_Meta_L, kKey_Meta},
   {GDK_Meta_R, kKey_Meta},
   {GDK_Alt_L, kKey_Alt},
   {GDK_Alt_R, kKey_Alt},
   {GDK_Caps_Lock, kKey_CapsLock},
   {GDK_Num_Lock, kKey_NumLock},
   {GDK_Scroll_Lock, kKey_ScrollLock},
   {GDK_KP_Space, kKey_Space},  // numeric keypad
   {GDK_KP_Tab, kKey_Tab},
   {GDK_KP_Enter, kKey_Enter},
   {GDK_KP_Equal, kKey_Equal},
   {GDK_KP_Multiply, kKey_Asterisk},
   {GDK_KP_Add, kKey_Plus},
   {GDK_KP_Separator, kKey_Comma},
   {GDK_KP_Subtract, kKey_Minus},
   {GDK_KP_Decimal, kKey_Period},
   {GDK_KP_Divide, kKey_Slash},
   {0, (EKeySym) 0}
};

//______________________________________________________________________________
static void _ChangeProperty(HWND w, char *np, char *dp, int n, Atom_t type)
{
   HGLOBAL hMem;
   char *p;

   hMem = GetProp(w, np);
   if (hMem != NULL) {
      GlobalFree(hMem);
   }
   hMem = GlobalAlloc(GHND, n + sizeof(Atom_t));
   p = (char *) GlobalLock(hMem);
   memcpy(p, &type, sizeof(Atom_t));
   memcpy(p + sizeof(Atom_t), dp, n);
   GlobalUnlock(hMem);
   SetProp(w, np, hMem);
}

//______________________________________________________________________________
void W32ChangeProperty(HWND w, Atom_t property, Atom_t type,
                       int format, int mode, const unsigned char *data,
                       int nelements)
{
   char *atomName;
   char buffer[256];
   char *p, *s;
   int len;
   char propName[8];

   if (mode == GDK_PROP_MODE_REPLACE || mode == GDK_PROP_MODE_PREPEND) {
      len = (int) GlobalGetAtomName(property, buffer, sizeof(buffer));
      if ((atomName = (char *) malloc(len + 1)) == NULL) {
         return;
      } else {
         strcpy(atomName, buffer);
      }
      sprintf(propName, "#0x%0.4x", atomName);
      _ChangeProperty(w, propName, (char *) data, nelements, type);
   }
}

//______________________________________________________________________________
int _GetWindowProperty(GdkWindow * id, Atom_t property, Long_t long_offset,
                       Long_t long_length, Bool_t delete_it, Atom_t req_type,
                       Atom_t * actual_type_return,
                       Int_t * actual_format_return, ULong_t * nitems_return,
                       ULong_t * bytes_after_return, UChar_t ** prop_return)
{
   char *atomName;
   char *data, *destPtr;
   char propName[8];
   HGLOBAL handle;
   HGLOBAL hMem;
   HWND w;

   w = (HWND) GDK_DRAWABLE_XID(id);

   if (IsClipboardFormatAvailable(CF_TEXT) && OpenClipboard(NULL)) {
      handle = GetClipboardData(CF_TEXT);
      if (handle != NULL) {
         data = (char *) GlobalLock(handle);
         *nitems_return = strlen(data);
         *prop_return = (UChar_t *) malloc(*nitems_return + 1);
         destPtr = (char *) *prop_return;
         while (*data != '\0') {
            if (*data != '\r') {
               *destPtr = *data;
               destPtr++;
            }
            data++;
         }
         *destPtr = '\0';
         GlobalUnlock(handle);
         *actual_type_return = XA_STRING;
         *bytes_after_return = 0;
      }
      CloseClipboard();
      return 1;
   }
   if (delete_it)
      RemoveProp(w, propName);
   return 1;
}

//______________________________________________________________________________
BOOL CALLBACK EnumChildProc(HWND hwndChild, LPARAM lParam)
{
   // Make sure the child window is visible.

   ShowWindow(hwndChild, SW_SHOWNORMAL);
//    SetForegroundWindow (hwndChild);
//    BringWindowToTop (hwndChild);

   return TRUE;
}

//______________________________________________________________________________
inline void SplitLong(Long_t ll, Long_t & i1, Long_t & i2)
{
   union {
      Long_t l;
      Int_t i[2];
   } conv;

   conv.l = ll;
   i1 = conv.i[0];
   i2 = conv.i[1];
}

//______________________________________________________________________________
inline void AsmLong(Long_t i1, Long_t i2, Long_t & ll)
{
   union {
      Long_t l;
      Int_t i[2];
   } conv;

   conv.i[0] = (Int_t) i1;
   conv.i[1] = (Int_t) i2;
   ll = conv.l;
}

//______________________________________________________________________________
void TGWin32::MapWindow(Window_t id)
{
   // Map window on screen.

   gdk_window_show((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::MapSubwindows(Window_t id)
{

   HWND wp;
   EnumChildWindows((HWND) GDK_DRAWABLE_XID((GdkWindow *) id),
                    EnumChildProc, (LPARAM) NULL);

}

//______________________________________________________________________________
void TGWin32::MapRaised(Window_t id)
{
   // Map window on screen and put on top of all windows.

   gdk_window_show((GdkWindow *) id);
   gdk_window_raise((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::UnmapWindow(Window_t id)
{
   // Unmap window from screen.
   gdk_window_hide((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::DestroyWindow(Window_t id)
{
   // Destroy window.

   gdk_window_destroy((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.

   gdk_window_raise((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.

   gdk_window_lower((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.

   gdk_window_move((GdkWindow *) id, x, y);
}

//______________________________________________________________________________
void TGWin32::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w,
                               UInt_t h)
{
   // Move and resize a window.

   gdk_window_move_resize((GdkWindow *) id, x, y, w, h);
}

//______________________________________________________________________________
void TGWin32::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.

   gdk_window_resize((GdkWindow *) id, w, h);
}

//______________________________________________________________________________
void TGWin32::IconifyWindow(Window_t id)
{
   // Iconify the window.

   gdk_window_lower((GdkWindow *) id);
}

//______________________________________________________________________________
void TGWin32::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.
   GdkColor back;
   back.pixel = color;
   back.red = GetRValue(color);
   back.green = GetGValue(color);
   back.blue = GetBValue(color);

   gdk_window_set_background((GdkWindow *) id, &back);
}

//______________________________________________________________________________
void TGWin32::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.

   gdk_window_set_back_pixmap((GdkWindow *) id, (GdkPixmap *) pxm, 0);
}

//______________________________________________________________________________
Window_t TGWin32::CreateWindow(Window_t parent, Int_t x, Int_t y,
                               UInt_t w, UInt_t h, UInt_t border,
                               Int_t depth, UInt_t clss,
                               void *visual, SetWindowAttributes_t * attr,
                               UInt_t wtype)
{
   // Return handle to newly created gdk window.

   GdkWMDecoration deco;
   GdkWindowAttr xattr;
   GdkWindow *newWin;
   GdkColor background_color;
   ULong_t xmask = 0;
   UInt_t xevmask;

   if (attr) {
      MapSetWindowAttributes(attr, xmask, xattr);
      xattr.window_type = GDK_WINDOW_CHILD;
      if (wtype & kTransientFrame) {
         xattr.window_type = GDK_WINDOW_DIALOG;
      }
      if (wtype & kMainFrame) {
         xattr.window_type = GDK_WINDOW_TOPLEVEL;
      }
      if (wtype & kTempFrame) {
         xattr.window_type = GDK_WINDOW_TEMP;
      }
      newWin = gdk_window_new((GdkWindow *) parent, &xattr, xmask);
   } else {
      xattr.width = w;
      xattr.height = h;
      xattr.wclass = GDK_INPUT_OUTPUT;
      xattr.event_mask = 0L;    //GDK_ALL_EVENTS_MASK;
      xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
          GDK_PROPERTY_CHANGE_MASK;
//                          GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK;
      if (x >= 0)
         xattr.x = x;
      else
         xattr.x = -1.0 * x;
      if (y >= 0)
         xattr.y = y;
      else
         xattr.y = -1.0 * y;
      xattr.colormap = gdk_colormap_get_system();
      xattr.cursor = NULL;
      xattr.override_redirect = TRUE;
      if ((xattr.y > 0) && (xattr.x > 0))
         xmask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP |
             GDK_WA_WMCLASS | GDK_WA_NOREDIR;
      else
         xmask = GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
      if (visual != NULL) {
         xattr.visual = (GdkVisual *) visual;
         xmask |= GDK_WA_VISUAL;
      } else {
         xattr.visual = gdk_visual_get_system();
         xmask |= GDK_WA_VISUAL;
      }
      xattr.window_type = GDK_WINDOW_CHILD;
      if (wtype & kTransientFrame) {
         xattr.window_type = GDK_WINDOW_DIALOG;
      }
      if (wtype & kMainFrame) {
         xattr.window_type = GDK_WINDOW_TOPLEVEL;
      }
      if (wtype & kTempFrame) {
         xattr.window_type = GDK_WINDOW_TEMP;
      }
      newWin = gdk_window_new((GdkWindow *) parent, &xattr, xmask);
      gdk_window_set_events(newWin, (GdkEventMask) 0L);
   }
   if (border > 0) {
      gdk_window_set_decorations(newWin,
                                 (GdkWMDecoration) GDK_DECOR_BORDER);
   }
   if (attr) {
      if ((attr->fMask & kWABackPixmap)) {
         if (attr->fBackgroundPixmap == kNone) {
            gdk_window_set_back_pixmap(newWin, (GdkPixmap *) GDK_NONE, 0);
         } else if (attr->fBackgroundPixmap == kParentRelative) {
            gdk_window_set_back_pixmap(newWin, (GdkPixmap *) GDK_NONE, 1);
         } else {
            gdk_window_set_back_pixmap(newWin,
                                       (GdkPixmap *) attr->
                                       fBackgroundPixmap, 0);
         }
      }
      if ((attr->fMask & kWABackPixel)) {
         background_color.pixel = attr->fBackgroundPixel;
         background_color.red = GetRValue(attr->fBackgroundPixel);
         background_color.green = GetGValue(attr->fBackgroundPixel);
         background_color.blue = GetBValue(attr->fBackgroundPixel);
         gdk_window_set_background(newWin, &background_color);
      }
   }
   return (Window_t) newWin;

}

//______________________________________________________________________________
void TGWin32::MapEventMask(UInt_t & emask, UInt_t & xemask, Bool_t tox)
{
   // Map event mask to or from gdk.

   if (tox) {
      Long_t lxemask = 0L;
      if ((emask & kKeyPressMask))
         lxemask |= GDK_KEY_PRESS_MASK;
      if ((emask & kKeyReleaseMask))
         lxemask |= GDK_KEY_RELEASE_MASK;
      if ((emask & kButtonPressMask))
         lxemask |= GDK_BUTTON_PRESS_MASK;
      if ((emask & kButtonReleaseMask))
         lxemask |= GDK_BUTTON_RELEASE_MASK;
      if ((emask & kPointerMotionMask))
         lxemask |= GDK_POINTER_MOTION_MASK;
      if ((emask & kButtonMotionMask))
         lxemask |= GDK_BUTTON_MOTION_MASK;
      if ((emask & kExposureMask))
         lxemask |= GDK_EXPOSURE_MASK;
      if ((emask & kStructureNotifyMask))
         lxemask |= GDK_STRUCTURE_MASK;
      if ((emask & kEnterWindowMask))
         lxemask |= GDK_ENTER_NOTIFY_MASK;
      if ((emask & kLeaveWindowMask))
         lxemask |= GDK_LEAVE_NOTIFY_MASK;
      if ((emask & kFocusChangeMask))
         lxemask |= GDK_FOCUS_CHANGE_MASK;
      xemask = (UInt_t) lxemask;
   } else {
      emask = 0;
      if ((xemask & GDK_KEY_PRESS_MASK))
         emask |= kKeyPressMask;
      if ((xemask & GDK_KEY_RELEASE_MASK))
         emask |= kKeyReleaseMask;
      if ((xemask & GDK_BUTTON_PRESS_MASK))
         emask |= kButtonPressMask;
      if ((xemask & GDK_BUTTON_RELEASE_MASK))
         emask |= kButtonReleaseMask;
      if ((xemask & GDK_POINTER_MOTION_MASK))
         emask |= kPointerMotionMask;
      if ((xemask & GDK_BUTTON_MOTION_MASK))
         emask |= kButtonMotionMask;
      if ((xemask & GDK_EXPOSURE_MASK))
         emask |= kExposureMask;
      if ((xemask & GDK_STRUCTURE_MASK))
         emask |= kStructureNotifyMask;
      if ((xemask & GDK_ENTER_NOTIFY_MASK))
         emask |= kEnterWindowMask;
      if ((xemask & GDK_LEAVE_NOTIFY_MASK))
         emask |= kLeaveWindowMask;
      if ((xemask & GDK_FOCUS_CHANGE_MASK))
         emask |= kFocusChangeMask;
   }
}

//______________________________________________________________________________
void TGWin32::MapSetWindowAttributes(SetWindowAttributes_t * attr,
                                     ULong_t & xmask,
                                     GdkWindowAttr & xattr)
{
   // Map a SetWindowAttributes_t to a GdkWindowAttr structure.

   Mask_t mask = attr->fMask;
   xmask = 0;

   if ((mask & kWAOverrideRedirect)) {
      xmask |= GDK_WA_NOREDIR;
      xattr.override_redirect = attr->fOverrideRedirect;
   }
   if ((mask & kWAEventMask)) {
      UInt_t xmsk, msk = (UInt_t) attr->fEventMask;
      MapEventMask(msk, xmsk, kTRUE);
      xattr.event_mask = xmsk;
   }
   if ((mask & kWAColormap)) {
      xmask |= GDK_WA_COLORMAP;
      xattr.colormap = (GdkColormap *) attr->fColormap;
   }
   if ((mask & kWACursor)) {
      xmask |= GDK_WA_CURSOR;
      if (attr->fCursor != kNone)
         xattr.cursor = (GdkCursor *) attr->fCursor;
   }
   xattr.wclass = GDK_INPUT_OUTPUT;

}

//______________________________________________________________________________
void TGWin32::MapGCValues(GCValues_t & gval,
                          ULong_t & xmask, GdkGCValues & xgval, Bool_t tox)
{
   // Map a GCValues_t to a XCGValues structure if tox is true. Map
   // the other way in case tox is false.

   if (tox) {
      // map GCValues_t to XGCValues
      Mask_t mask = gval.fMask;
      xmask = 0;

      if ((mask & kGCFunction)) {
         xmask |= GDK_GC_FUNCTION;
         switch (gval.fFunction) {
         case kGXclear:
            xgval.function = GDK_CLEAR;
            break;
         case kGXand:
            xgval.function = GDK_AND;
            break;
         case kGXandReverse:
            xgval.function = GDK_AND_REVERSE;
            break;
         case kGXcopy:
            xgval.function = GDK_COPY;
            break;
         case kGXandInverted:
            xgval.function = GDK_AND_INVERT;
            break;
         case kGXnoop:
            xgval.function = GDK_NOOP;
            break;
         case kGXxor:
            xgval.function = GDK_XOR;
            break;
         case kGXor:
            xgval.function = GDK_OR;
            break;
         case kGXequiv:
            xgval.function = GDK_EQUIV;
            break;
         case kGXinvert:
            xgval.function = GDK_INVERT;
            break;
         case kGXorReverse:
            xgval.function = GDK_OR_REVERSE;
            break;
         case kGXcopyInverted:
            xgval.function = GDK_COPY_INVERT;
            break;
         case kGXorInverted:
            xgval.function = GDK_OR_INVERT;
            break;
         case kGXnand:
            xgval.function = GDK_NAND;
            break;
         case kGXset:
            xgval.function = GDK_SET;
            break;
         }
      }
      if (mask & kGCSubwindowMode) {
         xmask |= GDK_GC_SUBWINDOW;
         if (gval.fSubwindowMode == kIncludeInferiors)
            xgval.subwindow_mode = GDK_INCLUDE_INFERIORS;
         else
            xgval.subwindow_mode = GDK_CLIP_BY_CHILDREN;
      }
      if ((mask & kGCForeground)) {
         xmask |= GDK_GC_FOREGROUND;
         xgval.foreground.pixel = gval.fForeground;
         xgval.foreground.red = GetRValue(gval.fForeground);
         xgval.foreground.green = GetGValue(gval.fForeground);
         xgval.foreground.blue = GetBValue(gval.fForeground);
      }
      if ((mask & kGCBackground)) {
         xmask |= GDK_GC_BACKGROUND;
         xgval.background.pixel = gval.fBackground;
         xgval.background.red = GetRValue(gval.fBackground);
         xgval.background.green = GetGValue(gval.fBackground);
         xgval.background.blue = GetBValue(gval.fBackground);
      }
      if ((mask & kGCLineWidth)) {
         xmask |= GDK_GC_LINE_WIDTH;
         xgval.line_width = gval.fLineWidth;
      }
      if ((mask & kGCLineStyle)) {
         xmask |= GDK_GC_LINE_STYLE;
         xgval.line_style = (GdkLineStyle) gval.fLineStyle;	// ident mapping
      }
      if ((mask & kGCCapStyle)) {
         xmask |= GDK_GC_CAP_STYLE;
         xgval.cap_style = (GdkCapStyle) gval.fCapStyle;	// ident mapping
      }
      if ((mask & kGCJoinStyle)) {
         xmask |= GDK_GC_JOIN_STYLE;
         xgval.join_style = (GdkJoinStyle) gval.fJoinStyle;	// ident mapping
      }
      if ((mask & kGCFillStyle)) {
         xmask |= GDK_GC_FILL;
         xgval.fill = (GdkFill) gval.fFillStyle;	// ident mapping
      }
      if ((mask & kGCTile)) {
         xmask |= GDK_GC_TILE;
         xgval.tile = (GdkPixmap *) gval.fTile;
      }
      if ((mask & kGCStipple)) {
         xmask |= GDK_GC_STIPPLE;
         xgval.stipple = (GdkPixmap *) gval.fStipple;
      }
      if ((mask & kGCTileStipXOrigin)) {
         xmask |= GDK_GC_TS_X_ORIGIN;
         xgval.ts_x_origin = gval.fTsXOrigin;
      }
      if ((mask & kGCTileStipYOrigin)) {
         xmask |= GDK_GC_TS_Y_ORIGIN;
         xgval.ts_y_origin = gval.fTsYOrigin;
      }
      if ((mask & kGCFont)) {
         xmask |= GDK_GC_FONT;
         xgval.font = (GdkFont *) gval.fFont;
      }
      if ((mask & kGCGraphicsExposures)) {
         xmask |= GDK_GC_EXPOSURES;
         xgval.graphics_exposures = gval.fGraphicsExposures;
      }
      if ((mask & kGCClipXOrigin)) {
         xmask |= GDK_GC_CLIP_X_ORIGIN;
         xgval.clip_x_origin = gval.fClipXOrigin;
      }
      if ((mask & kGCClipYOrigin)) {
         xmask |= GDK_GC_CLIP_Y_ORIGIN;
         xgval.clip_y_origin = gval.fClipYOrigin;
      }
      if ((mask & kGCClipMask)) {
         xmask |= GDK_GC_CLIP_MASK;
         xgval.clip_mask = (GdkPixmap *) gval.fClipMask;
      }

   } else {
      // map XValues to GCValues_t
      Mask_t mask = 0;

      if ((xmask & GDK_GC_FUNCTION)) {
         mask |= kGCFunction;
         gval.fFunction = (EGraphicsFunction) xgval.function;	// ident mapping
         switch (xgval.function) {
         case GDK_CLEAR:
            gval.fFunction = kGXclear;
            break;
         case GDK_AND:
            gval.fFunction = kGXand;
            break;
         case GDK_AND_REVERSE:
            gval.fFunction = kGXandReverse;
            break;
         case GDK_COPY:
            gval.fFunction = kGXcopy;
            break;
         case GDK_AND_INVERT:
            gval.fFunction = kGXandInverted;
            break;
         case GDK_NOOP:
            gval.fFunction = kGXnoop;
            break;
         case GDK_XOR:
            gval.fFunction = kGXxor;
            break;
         case GDK_OR:
            gval.fFunction = kGXor;
            break;
         case GDK_EQUIV:
            gval.fFunction = kGXequiv;
            break;
         case GDK_INVERT:
            gval.fFunction = kGXinvert;
            break;
         case GDK_OR_REVERSE:
            gval.fFunction = kGXorReverse;
            break;
         case GDK_COPY_INVERT:
            gval.fFunction = kGXcopyInverted;
            break;
         case GDK_OR_INVERT:
            gval.fFunction = kGXorInverted;
            break;
         case GDK_NAND:
            gval.fFunction = kGXnand;
            break;
         case GDK_SET:
            gval.fFunction = kGXset;
            break;
         }
      }
      if (xmask & GDK_GC_SUBWINDOW) {
         mask |= kGCSubwindowMode;
         if (xgval.subwindow_mode == GDK_INCLUDE_INFERIORS)
            gval.fSubwindowMode = kIncludeInferiors;
         else
            gval.fSubwindowMode = kClipByChildren;
      }
      if ((xmask & GDK_GC_FOREGROUND)) {
         mask |= kGCForeground;
         gval.fForeground = xgval.foreground.pixel;
      }
      if ((xmask & GDK_GC_BACKGROUND)) {
         mask |= kGCBackground;
         gval.fBackground = xgval.background.pixel;
      }
      if ((xmask & GDK_GC_LINE_WIDTH)) {
         mask |= kGCLineWidth;
         gval.fLineWidth = xgval.line_width;
      }
      if ((xmask & GDK_GC_LINE_STYLE)) {
         mask |= kGCLineStyle;
         gval.fLineStyle = xgval.line_style;	// ident mapping
      }
      if ((xmask & GDK_GC_CAP_STYLE)) {
         mask |= kGCCapStyle;
         gval.fCapStyle = xgval.cap_style;	// ident mapping
      }
      if ((xmask & GDK_GC_JOIN_STYLE)) {
         mask |= kGCJoinStyle;
         gval.fJoinStyle = xgval.join_style;	// ident mapping
      }
      if ((xmask & GDK_GC_FILL)) {
         mask |= kGCFillStyle;
         gval.fFillStyle = xgval.fill;	// ident mapping
      }
      if ((xmask & GDK_GC_TILE)) {
         mask |= kGCTile;
         gval.fTile = (Pixmap_t) xgval.tile;
      }
      if ((xmask & GDK_GC_STIPPLE)) {
         mask |= kGCStipple;
         gval.fStipple = (Pixmap_t) xgval.stipple;
      }
      if ((xmask & GDK_GC_TS_X_ORIGIN)) {
         mask |= kGCTileStipXOrigin;
         gval.fTsXOrigin = xgval.ts_x_origin;
      }
      if ((xmask & GDK_GC_TS_Y_ORIGIN)) {
         mask |= kGCTileStipYOrigin;
         gval.fTsYOrigin = xgval.ts_y_origin;
      }
      if ((xmask & GDK_GC_FONT)) {
         mask |= kGCFont;
         gval.fFont = (FontH_t) xgval.font;
      }
      if ((xmask & GDK_GC_EXPOSURES)) {
         mask |= kGCGraphicsExposures;
         gval.fGraphicsExposures = (Bool_t) xgval.graphics_exposures;
      }
      if ((xmask & GDK_GC_CLIP_X_ORIGIN)) {
         mask |= kGCClipXOrigin;
         gval.fClipXOrigin = xgval.clip_x_origin;
      }
      if ((xmask & GDK_GC_CLIP_Y_ORIGIN)) {
         mask |= kGCClipYOrigin;
         gval.fClipYOrigin = xgval.clip_y_origin;
      }
      if ((xmask & GDK_GC_CLIP_MASK)) {
         mask |= kGCClipMask;
         gval.fClipMask = (Pixmap_t) xgval.clip_mask;
      }
      gval.fMask = mask;
   }
}

//______________________________________________________________________________
void TGWin32::GetWindowAttributes(Window_t id, WindowAttributes_t & attr)
{
   // Get window attributes and return filled in attributes structure.

   GdkWindowAttr xattr;

   gdk_window_get_geometry((GdkWindow *) id,
                           &attr.fX,
                           &attr.fY,
                           &attr.fWidth, &attr.fHeight, &attr.fDepth);
   attr.fRoot = (Window_t) GDK_ROOT_PARENT();
   attr.fColormap = (Colormap_t) gdk_window_get_colormap((GdkWindow *) id);
   attr.fBorderWidth = 0;
   attr.fVisual = gdk_window_get_visual((GdkWindow *) id);
   attr.fClass = kInputOutput;
   attr.fBackingStore = kNotUseful;
   attr.fSaveUnder = kFALSE;
   attr.fMapInstalled = kTRUE;
   attr.fOverrideRedirect = kFALSE;   // boolean value for override-redirect
   if (!gdk_window_is_visible((GdkWindow *) id))
      attr.fMapState = kIsUnmapped;
   else if (!gdk_window_is_viewable((GdkWindow *) id))
      attr.fMapState = kIsUnviewable;
   else
      attr.fMapState = kIsViewable;
}

//______________________________________________________________________________
Int_t TGWin32::OpenDisplay(const char *dpyName)
{
   // Open connection to display server (if such a thing exist on the
   // current platform). On X11 this method returns on success the X
   // display socket descriptor (> 0), 0 in case of batch mode and < 0
   // in case of failure (cannot connect to display dpyName). It also
   // initializes the TGWin32 class via Init(). Called from TGClient ctor.

   if (gDebug <= 4)
      gdk_debug_level = gDebug;
   else
      gdk_debug_level = 0;

   // Init the GX11 class, sets a.o. fDisplay.
   if (!Init())
      return -1;
   return 1;
}

//______________________________________________________________________________
void TGWin32::CloseDisplay()
{

}

//______________________________________________________________________________
Display_t TGWin32::GetDisplay()
{
   return 0;
}

//______________________________________________________________________________
Atom_t TGWin32::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.

   GdkAtom a = gdk_atom_intern((const gchar *) atom_name, only_if_exist);
   if (a == None)
      return kNone;
   return (Atom_t) a;
}

//______________________________________________________________________________
Window_t TGWin32::GetDefaultRootWindow()
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().
   return (Window_t) GDK_ROOT_PARENT();

}

//______________________________________________________________________________
Window_t TGWin32::GetParent(Window_t id)
{
   // Return the parent of the window.

   GdkWindow *parent;

   parent = gdk_window_get_parent((GdkWindow *) id);

   return (Window_t) parent;
}


//______________________________________________________________________________
FontStruct_t TGWin32::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise an opaque pointer to the FontStruct_t.
   // Free the loaded font using DeleteFont().

   GdkFont *fs = gdk_font_load(font_name);
   return (FontStruct_t) fs;
}

//______________________________________________________________________________
FontH_t TGWin32::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.

   if (fs) {
      GdkFont *fss = gdk_font_ref((GdkFont *) fs);
      return (FontH_t) fss;
   }
   return 0;
}

//______________________________________________________________________________
void TGWin32::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure obtained with LoadQueryFont().

   gdk_font_unref((GdkFont *) fs);
}

//______________________________________________________________________________
GContext_t TGWin32::CreateGC(Drawable_t id, GCValues_t * gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).

   GdkGCValues xgval;
   ULong_t xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, xgval, kTRUE);
   xgval.subwindow_mode = GDK_CLIP_BY_CHILDREN;	// GDK_INCLUDE_INFERIORS;
   GdkGC *gc = gdk_gc_new_with_values((GdkDrawable *) id,
                                      &xgval, (GdkGCValuesMask) xmask);

   return (GContext_t) gc;
}

//______________________________________________________________________________
void TGWin32::ChangeGC(GContext_t gc, GCValues_t * gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.

   GdkGCValues xgval;
   ULong_t xmask = 0;
   Mask_t mask = 0;

   if (gval) {
      mask = gval->fMask;
      MapGCValues(*gval, xmask, xgval, kTRUE);
   }
   if (mask & kGCForeground)
      gdk_gc_set_foreground((GdkGC *) gc, &xgval.foreground);
   if (mask & kGCBackground)
      gdk_gc_set_background((GdkGC *) gc, &xgval.background);
   if (mask & kGCFont)
      gdk_gc_set_font((GdkGC *) gc, xgval.font);
   if (mask & kGCFunction)
      gdk_gc_set_function((GdkGC *) gc, xgval.function);
   if (mask & kGCFillStyle)
      gdk_gc_set_fill((GdkGC *) gc, xgval.fill);
   if (mask & kGCTile)
      gdk_gc_set_tile((GdkGC *) gc, xgval.tile);
   if (mask & kGCStipple)
      gdk_gc_set_stipple((GdkGC *) gc, xgval.stipple);
   if ((mask & kGCTileStipXOrigin) || (mask & kGCTileStipYOrigin))
      gdk_gc_set_ts_origin((GdkGC *) gc, xgval.ts_x_origin,
                           xgval.ts_y_origin);
   if ((mask & kGCClipXOrigin) || (mask & kGCClipYOrigin))
      gdk_gc_set_clip_origin((GdkGC *) gc, xgval.clip_x_origin,
                             xgval.clip_y_origin);
   if (mask & kGCClipMask)
      gdk_gc_set_clip_mask((GdkGC *) gc, xgval.clip_mask);
   if (mask & kGCGraphicsExposures)
      gdk_gc_set_exposures((GdkGC *) gc, xgval.graphics_exposures);
   if ((mask & kGCLineWidth) || (mask & kGCLineStyle) ||
       (mask & kGCCapStyle) || (mask & kGCJoinStyle))
      gdk_gc_set_line_attributes((GdkGC *) gc, xgval.line_width,
                                 xgval.line_style, xgval.cap_style,
                                 xgval.join_style);
   if (mask & kGCSubwindowMode)
      gdk_gc_set_subwindow((GdkGC *) gc, xgval.subwindow_mode);

}

//______________________________________________________________________________
void TGWin32::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. Both org and dest must exist.

   GCValues_t gval;
   GdkGCValues xgval;
   ULong_t xmask;

   if (!mask) {
      // in this case copy all fields
      mask = (Mask_t) - 1;
   }

   gval.fMask = mask;           // only set fMask used to convert to xmask
   MapGCValues(gval, xmask, xgval, kTRUE);

   gdk_gc_copy((GdkGC *) dest, (GdkGC *) org);
}

//______________________________________________________________________________
void TGWin32::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.

   gdk_gc_unref((GdkGC *) gc);
}

//______________________________________________________________________________
Cursor_t TGWin32::CreateCursor(ECursor cursor)
{
   // Create cursor handle (just return cursor from cursor pool fCursors).

   return (Cursor_t) & fCursors[cursor];
}

//______________________________________________________________________________
void TGWin32::SetCursor(Window_t id, Cursor_t curid)
{
   // Set the specified cursor.

   gdk_window_set_cursor((GdkWindow *) id, (GdkCursor *) curid);
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.

   return (Pixmap_t) gdk_pixmap_new((GdkWindow *) id, w, h,
                                    gdk_visual_get_best_depth());
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreatePixmap(Drawable_t id, const char *bitmap,
                               UInt_t width, UInt_t height,
                               ULong_t forecolor, ULong_t backcolor,
                               Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.

   GdkColor fore, back;
   fore.pixel = forecolor;
   fore.red = GetRValue(forecolor);
   fore.green = GetGValue(forecolor);
   fore.blue = GetBValue(forecolor);

   back.pixel = backcolor;
   back.red = GetRValue(backcolor);
   back.green = GetGValue(backcolor);
   back.blue = GetBValue(backcolor);
   return (Pixmap_t) gdk_pixmap_create_from_data((GdkWindow *) id,
                                                 (char *) bitmap, width,
                                                 height, depth, &fore,
                                                 &back);
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreateBitmap(Drawable_t id, const char *bitmap,
                               UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.

   return (Pixmap_t) gdk_bitmap_create_from_data((GdkWindow *) id,
                                                 (char *) bitmap, width,
                                                 height);
}

//______________________________________________________________________________
void TGWin32::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.

   gdk_pixmap_unref((GdkPixmap *) pmap);
}

//______________________________________________________________________________
Bool_t TGWin32::CreatePictureFromFile(Drawable_t id, const char *filename,
                                      Pixmap_t & pict,
                                      Pixmap_t & pict_mask,
                                      PictureAttributes_t & attr)
{
   // Create a picture pixmap from data on file. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

   GdkBitmap *gdk_pixmap_mask;
   pict = (Pixmap_t) gdk_pixmap_create_from_xpm((GdkWindow *) id,
                                                &gdk_pixmap_mask, 0,
                                                filename);
   pict_mask = (Pixmap_t) gdk_pixmap_mask;
   gdk_drawable_get_size((GdkPixmap *) pict, (int *) &attr.fWidth,
                         (int *) &attr.fHeight);
   if (pict) {
      return kTRUE;
   }
   if (pict_mask) {
      pict_mask = kNone;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGWin32::CreatePictureFromData(Drawable_t id, char **data,
                                      Pixmap_t & pict,
                                      Pixmap_t & pict_mask,
                                      PictureAttributes_t & attr)
{
   // Create a pixture pixmap from data. The picture attributes
   // are used for input and output. Returns kTRUE in case of success,
   // kFALSE otherwise. If mask does not exist it is set to kNone.

   GdkBitmap *gdk_pixmap_mask;

   pict = (Pixmap_t) gdk_pixmap_create_from_xpm_d((GdkWindow *) id,
                                                  &gdk_pixmap_mask, 0,
                                                  data);
   pict_mask = (Pixmap_t) gdk_pixmap_mask;

   if (pict) {
      return kTRUE;
   }
   if (pict_mask) {
      pict_mask = kNone;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGWin32::ReadPictureDataFromFile(const char *filename,
                                        char ***ret_data)
{
   // Read picture data from file and store in ret_data. Returns kTRUE in
   // case of success, kFALSE otherwise.
//   GdkPixmap* rdata;
   char **rdata;
   rdata =
       (char **) gdk_pixmap_create_from_xpm(NULL, NULL, NULL, filename);
   ret_data = &rdata;
   return kTRUE;
   return kFALSE;
}

//______________________________________________________________________________
void TGWin32::DeletePictureData(void *data)
{
   // Delete picture data created by the function ReadPictureDataFromFile.

   free(data);
}

//______________________________________________________________________________
void TGWin32::SetDashes(GContext_t gc, Int_t offset, const char *dash_list,
                        Int_t n)
{
   // Specify a dash pattertn. Offset defines the phase of the pattern.
   // Each element in the dash_list array specifies the length (in pixels)
   // of a segment of the pattern. N defines the length of the list.
   int i;
   gint8 dashes[32];
   for (i = 0; i < n; i++)
      dashes[i] = (gint8) dash_list[i];
   for (i = n; i < 32; i++)
      dashes[i] = (gint8) 0;
   gdk_gc_set_dashes((GdkGC *) gc, offset, dashes, n);
}

//______________________________________________________________________________
void TGWin32::MapColorStruct(ColorStruct_t * color, GdkColor & xcolor)
{
   // Map a ColorStruct_t to a XColor structure.

   xcolor.pixel = color->fPixel;
   xcolor.red = color->fRed;
   xcolor.green = color->fGreen;
   xcolor.blue = color->fBlue;

}

//______________________________________________________________________________
Bool_t TGWin32::ParseColor(Colormap_t cmap, const char *cname,
                           ColorStruct_t & color)
{
   // Parse string cname containing color name, like "green" or "#00FF00".
   // It returns a filled in ColorStruct_t. Returns kFALSE in case parsing
   // failed, kTRUE in case of success. On success, the ColorStruct_t
   // fRed, fGreen and fBlue fields are all filled in and the mask is set
   // for all three colors, but fPixel is not set.

   GdkColor xc;

   if (gdk_color_parse((char *) cname, &xc)) {
      color.fPixel = xc.pixel = RGB(xc.red, xc.green, xc.blue);
      color.fRed = xc.red;
      color.fGreen = xc.green;
      color.fBlue = xc.blue;
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGWin32::AllocColor(Colormap_t cmap, ColorStruct_t & color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.

   int status;
   GdkColor xc;

   xc.red = color.fRed;
   xc.green = color.fGreen;
   xc.blue = color.fBlue;

   status = gdk_colormap_alloc_color((GdkColormap *) cmap,
                                     &xc, FALSE, TRUE);
   color.fPixel = xc.pixel;

   return kTRUE;                // status != 0 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGWin32::QueryColor(Colormap_t cmap, ColorStruct_t & color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.

   GdkColor xc;

   xc.pixel = color.fPixel;

   GdkColorContext *cc =
       gdk_color_context_new(gdk_visual_get_system(), fColormap);
   gdk_color_context_query_color(cc, &xc);

   color.fPixel = xc.pixel = RGB(xc.red, xc.green, xc.blue);
   color.fRed = xc.red;
   color.fGreen = xc.green;
   color.fBlue = xc.blue;
}

//______________________________________________________________________________
Int_t TGWin32::EventsPending()
{
   // Returns number of pending events.

   return gdk_events_pending();
}

//______________________________________________________________________________
void TGWin32::NextEvent(Event_t & event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.

   GdkEvent *xev = NULL;

   xev = gdk_event_get();

   // fill in Event_t
   event.fType = kOtherEvent;   // bb add
   if (xev == NULL)
      return;
   MapEvent(event, *xev, kFALSE);
//   gdk_event_free (xev);

}

//______________________________________________________________________________
void TGWin32::MapModifierState(UInt_t & state, UInt_t & xstate, Bool_t tox)
{
   // Map modifier key state to or from X.

   if (tox) {
      xstate = 0;
      if ((state & kKeyShiftMask))
         xstate |= GDK_SHIFT_MASK;
      if ((state & kKeyLockMask))
         xstate |= GDK_LOCK_MASK;
      if ((state & kKeyControlMask))
         xstate |= GDK_CONTROL_MASK;
      if ((state & kKeyMod1Mask))
         xstate |= GDK_MOD1_MASK;
      if ((state & kButton1Mask))
         xstate |= GDK_BUTTON1_MASK;
      if ((state & kButton2Mask))
         xstate |= GDK_BUTTON2_MASK;
      if ((state & kButton3Mask))
         xstate |= GDK_BUTTON3_MASK;
      if ((state & kAnyModifier))
         xstate |= GDK_MODIFIER_MASK;	// or should it be = instead of |= ?
   } else {
      state = 0;
      if ((xstate & GDK_SHIFT_MASK))
         state |= kKeyShiftMask;
      if ((xstate & GDK_LOCK_MASK))
         state |= kKeyLockMask;
      if ((xstate & GDK_CONTROL_MASK))
         state |= kKeyControlMask;
      if ((xstate & GDK_MOD1_MASK))
         state |= kKeyMod1Mask;
      if ((xstate & GDK_BUTTON1_MASK))
         state |= kButton1Mask;
      if ((xstate & GDK_BUTTON2_MASK))
         state |= kButton2Mask;
      if ((xstate & GDK_BUTTON3_MASK))
         state |= kButton3Mask;
      if ((xstate & GDK_MODIFIER_MASK))
         state |= kAnyModifier; // idem
   }
}

void _set_event_time(GdkEvent * event, UInt_t time)
{
   if (event)
      switch (event->type) {
      case GDK_MOTION_NOTIFY:
         event->motion.time = time;
      case GDK_BUTTON_PRESS:
      case GDK_2BUTTON_PRESS:
      case GDK_3BUTTON_PRESS:
      case GDK_BUTTON_RELEASE:
      case GDK_SCROLL:
         event->button.time = time;
      case GDK_KEY_PRESS:
      case GDK_KEY_RELEASE:
         event->key.time = time;
      case GDK_ENTER_NOTIFY:
      case GDK_LEAVE_NOTIFY:
         event->crossing.time = time;
      case GDK_PROPERTY_NOTIFY:
         event->property.time = time;
      case GDK_SELECTION_CLEAR:
      case GDK_SELECTION_REQUEST:
      case GDK_SELECTION_NOTIFY:
         event->selection.time = time;
      case GDK_PROXIMITY_IN:
      case GDK_PROXIMITY_OUT:
         event->proximity.time = time;
      case GDK_DRAG_ENTER:
      case GDK_DRAG_LEAVE:
      case GDK_DRAG_MOTION:
      case GDK_DRAG_STATUS:
      case GDK_DROP_START:
      case GDK_DROP_FINISHED:
         event->dnd.time = time;
      default:                 /* use current time */
         break;
      }
}

//______________________________________________________________________________
void TGWin32::MapEvent(Event_t & ev, GdkEvent & xev, Bool_t tox)
{
   // Map Event_t structure to gdk_event structure. If tox is false
   // map the other way.

   if (tox) {
      // map from Event_t to gdk_event
      xev.type = GDK_NOTHING;
      if (ev.fType == kGKeyPress)
         xev.type = GDK_KEY_PRESS;
      if (ev.fType == kKeyRelease)
         xev.type = GDK_KEY_RELEASE;
      if (ev.fType == kButtonPress)
         xev.type = GDK_BUTTON_PRESS;
      if (ev.fType == kButtonRelease)
         xev.type = GDK_BUTTON_RELEASE;
      if (ev.fType == kMotionNotify)
         xev.type = GDK_MOTION_NOTIFY;
      if (ev.fType == kEnterNotify)
         xev.type = GDK_ENTER_NOTIFY;
      if (ev.fType == kLeaveNotify)
         xev.type = GDK_LEAVE_NOTIFY;
      if (ev.fType == kExpose)
         xev.type = GDK_EXPOSE;
      if (ev.fType == kConfigureNotify)
         xev.type = GDK_CONFIGURE;
      if (ev.fType == kMapNotify)
         xev.type = GDK_MAP;
      if (ev.fType == kUnmapNotify)
         xev.type = GDK_UNMAP;
      if (ev.fType == kDestroyNotify)
         xev.type = GDK_DESTROY;
      if (ev.fType == kClientMessage)
         xev.type = GDK_CLIENT_EVENT;
      if (ev.fType == kSelectionClear)
         xev.type = GDK_SELECTION_CLEAR;
      if (ev.fType == kSelectionRequest)
         xev.type = GDK_SELECTION_REQUEST;
      if (ev.fType == kSelectionNotify)
         xev.type = GDK_SELECTION_NOTIFY;

      xev.any.type = xev.type;
      xev.any.send_event = ev.fSendEvent;
      if (ev.fType == kDestroyNotify) {
         xev.any.window = (GdkWindow *) ev.fWindow;
      }
      if (ev.fType == kFocusIn) {
         xev.type = GDK_FOCUS_CHANGE;
         xev.focus_change.type = xev.type;
         xev.focus_change.window = (GdkWindow *) ev.fWindow;
         xev.focus_change.in = TRUE;
      }
      if (ev.fType == kFocusOut) {
         xev.type = GDK_FOCUS_CHANGE;
         xev.focus_change.type = xev.type;
         xev.focus_change.window = (GdkWindow *) ev.fWindow;
         xev.focus_change.in = FALSE;
      }
      if (ev.fType == kGKeyPress || ev.fType == kKeyRelease) {
         xev.key.window = (GdkWindow *) ev.fWindow;
         xev.key.type = xev.type;
         MapModifierState(ev.fState, xev.key.state, kTRUE);	// key mask
         xev.key.keyval = ev.fCode;	// key code
      }
      if (ev.fType == kButtonPress || ev.fType == kButtonRelease) {
         xev.button.window = (GdkWindow *) ev.fWindow;
         xev.button.type = xev.type;
         xev.button.x = ev.fX;
         xev.button.y = ev.fY;
         xev.button.x_root = ev.fXRoot;
         xev.button.y_root = ev.fYRoot;
         MapModifierState(ev.fState, xev.button.state, kTRUE);	// button mask
         if (ev.fType == kButtonRelease)
            xev.button.state |= GDK_RELEASE_MASK;
         xev.button.button = ev.fCode;	// button code
      }
      if (ev.fType == kSelectionNotify) {
         xev.selection.window = (GdkWindow *) ev.fUser[0];
         xev.selection.requestor = (guint32) ev.fUser[0];
         xev.selection.selection = (GdkAtom) ev.fUser[1];
         xev.selection.target = (GdkAtom) ev.fUser[2];
         xev.selection.property = (GdkAtom) ev.fUser[3];
         xev.selection.type = xev.type;
      }
      if (ev.fType == kClientMessage) {
         if ((ev.fFormat == 32) && (ev.fHandle == gWM_DELETE_WINDOW)) {
            xev.type = GDK_DELETE;
            xev.any.type = xev.type;
            xev.any.window = (GdkWindow *) ev.fWindow;
         } else {
            xev.client.window = (GdkWindow *) ev.fWindow;
            xev.client.type = xev.type;
            xev.client.message_type = (GdkAtom) ev.fHandle;
            xev.client.data_format = ev.fFormat;
            xev.client.data.l[0] = ev.fUser[0];
            if (sizeof(ev.fUser[0]) > 4) {
               SplitLong(ev.fUser[1], xev.client.data.l[1],
                         xev.client.data.l[3]);
               SplitLong(ev.fUser[2], xev.client.data.l[2],
                         xev.client.data.l[4]);
            } else {
               xev.client.data.l[1] = ev.fUser[1];
               xev.client.data.l[2] = ev.fUser[2];
               xev.client.data.l[3] = ev.fUser[3];
               xev.client.data.l[4] = ev.fUser[4];
            }
         }
      }
      if (ev.fType == kMotionNotify) {
         xev.motion.window = (GdkWindow *) ev.fWindow;
         xev.motion.type = xev.type;
         xev.motion.x = ev.fX;
         xev.motion.y = ev.fY;
         xev.motion.x_root = ev.fXRoot;
         xev.motion.y_root = ev.fYRoot;
      }
      if ((ev.fType == kEnterNotify) || (ev.fType == kLeaveNotify)) {
         xev.crossing.window = (GdkWindow *) ev.fWindow;
         xev.crossing.type = xev.type;
         xev.crossing.x = ev.fX;
         xev.crossing.y = ev.fY;
         xev.crossing.x_root = ev.fXRoot;
         xev.crossing.y_root = ev.fYRoot;
         xev.crossing.mode = (GdkCrossingMode) ev.fCode;	// NotifyNormal, NotifyGrab, NotifyUngrab
         MapModifierState(ev.fState, xev.crossing.state, kTRUE);	// key or button mask
      }
      if (ev.fType == kExpose) {
         xev.expose.window = (GdkWindow *) ev.fWindow;
         xev.expose.type = xev.type;
         xev.expose.area.x = ev.fX;
         xev.expose.area.y = ev.fY;
         xev.expose.area.width = ev.fWidth;	// width and
         xev.expose.area.height = ev.fHeight;	// height of exposed area
         xev.expose.count = ev.fCount;	// number of expose events still to come
      }
      if (ev.fType == kConfigureNotify) {
         xev.configure.window = (GdkWindow *) ev.fWindow;
         xev.configure.type = xev.type;
         xev.configure.x = ev.fX;
         xev.configure.y = ev.fY;
         xev.configure.width = ev.fWidth;
         xev.configure.height = ev.fHeight;
      }
      if (ev.fType == kSelectionClear) {
         xev.selection.window = (GdkWindow *) ev.fWindow;
         xev.selection.type = xev.type;
         xev.selection.selection = ev.fUser[0];
      }
      if (ev.fType == kSelectionRequest) {
         xev.selection.window = (GdkWindow *) ev.fUser[0];
         xev.selection.type = xev.type;
         xev.selection.selection = ev.fUser[1];
         xev.selection.target = ev.fUser[2];
         xev.selection.property = ev.fUser[3];
      }
      if ((ev.fType == kMapNotify) || (ev.fType == kUnmapNotify)) {
         xev.any.window = (GdkWindow *) ev.fWindow;
      }
      _set_event_time(&xev, ev.fTime);
   } else {
      // map from gdk_event to Event_t
      ev.fType = kOtherEvent;
      if (xev.type == GDK_KEY_PRESS)
         ev.fType = kGKeyPress;
      if (xev.type == GDK_KEY_RELEASE)
         ev.fType = kKeyRelease;
      if (xev.type == GDK_BUTTON_PRESS)
         ev.fType = kButtonPress;
      if (xev.type == GDK_BUTTON_RELEASE)
         ev.fType = kButtonRelease;
      if (xev.type == GDK_MOTION_NOTIFY)
         ev.fType = kMotionNotify;
      if (xev.type == GDK_ENTER_NOTIFY)
         ev.fType = kEnterNotify;
      if (xev.type == GDK_LEAVE_NOTIFY)
         ev.fType = kLeaveNotify;
      if (xev.type == GDK_EXPOSE)
         ev.fType = kExpose;
      if (xev.type == GDK_CONFIGURE)
         ev.fType = kConfigureNotify;
      if (xev.type == GDK_MAP)
         ev.fType = kMapNotify;
      if (xev.type == GDK_UNMAP)
         ev.fType = kUnmapNotify;
      if (xev.type == GDK_DESTROY)
         ev.fType = kDestroyNotify;
      if (xev.type == GDK_SELECTION_CLEAR)
         ev.fType = kSelectionClear;
      if (xev.type == GDK_SELECTION_REQUEST)
         ev.fType = kSelectionRequest;
      if (xev.type == GDK_SELECTION_NOTIFY)
         ev.fType = kSelectionNotify;

      ev.fSendEvent = xev.any.send_event ? kTRUE : kFALSE;
      ev.fTime = gdk_event_get_time(&xev);

      if ((xev.type == GDK_MAP) || (xev.type == GDK_UNMAP)) {
         ev.fWindow = (Window_t) xev.any.window;
      }
      if (xev.type == GDK_DELETE) {
         ev.fWindow = (Window_t) xev.any.window;
         ev.fType = kClientMessage;
         ev.fFormat = 32;
         ev.fHandle = gWM_DELETE_WINDOW;
         ev.fUser[0] = (Long_t) gWM_DELETE_WINDOW;
         if (sizeof(ev.fUser[0]) > 4) {
            AsmLong(xev.client.data.l[1], xev.client.data.l[3],
                    ev.fUser[1]);
            AsmLong(xev.client.data.l[2], xev.client.data.l[4],
                    ev.fUser[2]);
         } else {
            ev.fUser[1] = xev.client.data.l[1];
            ev.fUser[2] = xev.client.data.l[2];
            ev.fUser[3] = xev.client.data.l[3];
            ev.fUser[4] = xev.client.data.l[4];
         }
      }
      if (xev.type == GDK_DESTROY) {
         ev.fType = kDestroyNotify;
         ev.fHandle = (Window_t) xev.any.window;	// window to be destroyed
         ev.fWindow = (Window_t) xev.any.window;
      }
      if (xev.type == GDK_FOCUS_CHANGE) {
         ev.fWindow = (Window_t) xev.focus_change.window;
         ev.fCode = kNotifyNormal;
         ev.fState = 0;
         if (xev.focus_change.in == TRUE)
            ev.fType = kFocusIn;
         else
            ev.fType = kFocusOut;
      }
      if (ev.fType == kGKeyPress || ev.fType == kKeyRelease) {
         ev.fWindow = (Window_t) xev.key.window;
         MapModifierState(ev.fState, xev.key.state, kFALSE);	// key mask
         ev.fCode = xev.key.keyval;	// key code
         ev.fUser[1] = xev.key.length;
         if (xev.key.length > 0)
            ev.fUser[2] = xev.key.string[0];
         if (xev.key.length > 1)
            ev.fUser[3] = xev.key.string[1];
         if (xev.key.length > 2)
            ev.fUser[4] = xev.key.string[2];
         HWND tmpwin =
             GetWindow((HWND) GDK_DRAWABLE_XID(xev.key.window), GW_CHILD);
         if (tmpwin)
            ev.fUser[0] = (ULong_t) gdk_xid_table_lookup(tmpwin);
         else
            ev.fUser[0] = (ULong_t) xev.key.window;
      }
      if (ev.fType == kButtonPress || ev.fType == kButtonRelease) {
         ev.fWindow = (Window_t) xev.button.window;
         ev.fX = xev.button.x;
         ev.fY = xev.button.y;
         ev.fXRoot = xev.button.x_root;
         ev.fYRoot = xev.button.y_root;
         MapModifierState(ev.fState, xev.button.state, kFALSE);	// button mask
         if (ev.fType == kButtonRelease)
            ev.fState |= kButtonReleaseMask;
         ev.fCode = xev.button.button;	// button code
         POINT tpoint;
         tpoint.x = xev.button.x;
         tpoint.y = xev.button.y;
         HWND tmpwin =
             ChildWindowFromPoint((HWND)
                                  GDK_DRAWABLE_XID(xev.button.window),
                                  tpoint);
         if (tmpwin)
            ev.fUser[0] = (ULong_t) gdk_xid_table_lookup(tmpwin);
         else
            ev.fUser[0] = (ULong_t) 0;
      }
      if (ev.fType == kMotionNotify) {
         ev.fWindow = (Window_t) xev.motion.window;
         ev.fX = xev.motion.x;
         ev.fY = xev.motion.y;
         ev.fXRoot = xev.motion.x_root;
         ev.fYRoot = xev.motion.y_root;
         MapModifierState(ev.fState, xev.motion.state, kFALSE);	// key or button mask
         POINT tpoint;
         tpoint.x = xev.button.x;
         tpoint.y = xev.button.y;
         HWND tmpwin =
             ChildWindowFromPoint((HWND)
                                  GDK_DRAWABLE_XID(xev.motion.window),
                                  tpoint);
         if (tmpwin)
            ev.fUser[0] = (ULong_t) gdk_xid_table_lookup(tmpwin);
         else
            ev.fUser[0] = (ULong_t) xev.motion.window;
      }
      if (ev.fType == kEnterNotify || ev.fType == kLeaveNotify) {
         ev.fWindow = (Window_t) xev.crossing.window;
         ev.fX = xev.crossing.x;
         ev.fY = xev.crossing.y;
         ev.fXRoot = xev.crossing.x_root;
         ev.fYRoot = xev.crossing.y_root;
         ev.fCode = xev.crossing.mode;	// NotifyNormal, NotifyGrab, NotifyUngrab
         MapModifierState(ev.fState, xev.crossing.state, kFALSE);	// key or button mask
      }
      if (ev.fType == kExpose) {
         ev.fWindow = (Window_t) xev.expose.window;
         ev.fX = xev.expose.area.x;
         ev.fY = xev.expose.area.y;
         ev.fWidth = xev.expose.area.width;	// width and
         ev.fHeight = xev.expose.area.height;	// height of exposed area
         ev.fCount = xev.expose.count;	// number of expose events still to come
      }
      if (ev.fType == kConfigureNotify) {
         ev.fWindow = (Window_t) xev.configure.window;
         ev.fX = xev.configure.x;
         ev.fY = xev.configure.y;
         ev.fWidth = xev.configure.width;
         ev.fHeight = xev.configure.height;
      }
      if (xev.type == GDK_CLIENT_EVENT) {
         ev.fWindow = (Window_t) xev.client.window;
         ev.fType = kClientMessage;
         ev.fHandle = xev.client.message_type;
         ev.fFormat = xev.client.data_format;
         ev.fUser[0] = xev.client.data.l[0];
         if (sizeof(ev.fUser[0]) > 4) {
            AsmLong(xev.client.data.l[1], xev.client.data.l[3],
                    ev.fUser[1]);
            AsmLong(xev.client.data.l[2], xev.client.data.l[4],
                    ev.fUser[2]);
         } else {
            ev.fUser[1] = xev.client.data.l[1];
            ev.fUser[2] = xev.client.data.l[2];
            ev.fUser[3] = xev.client.data.l[3];
            ev.fUser[4] = xev.client.data.l[4];
         }
      }
      if (ev.fType == kSelectionClear) {
         ev.fWindow = (Window_t) xev.selection.window;
         ev.fUser[0] = xev.selection.selection;
      }
      if (ev.fType == kSelectionRequest) {
         ev.fWindow = (Window_t) xev.selection.window;
         ev.fUser[0] = (ULong_t) xev.selection.window;
         ev.fUser[1] = xev.selection.selection;
         ev.fUser[2] = xev.selection.target;
         ev.fUser[3] = xev.selection.property;
      }
      if (ev.fType == kSelectionNotify) {
         ev.fWindow = (Window_t) xev.selection.window;
         ev.fUser[0] = (ULong_t) xev.selection.window;
         ev.fUser[1] = xev.selection.selection;
         ev.fUser[2] = xev.selection.target;
         ev.fUser[3] = xev.selection.property;
      }
   }
}

//______________________________________________________________________________
void TGWin32::Bell(Int_t percent)
{
   gdk_beep();
}

//______________________________________________________________________________
void TGWin32::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                       Int_t src_x, Int_t src_y, UInt_t width,
                       UInt_t height, Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.

   gdk_window_copy_area((GdkDrawable *) dest, (GdkGC *) gc, dest_x, dest_y,
                        (GdkDrawable *) src, src_x, src_y, width, height);
}

//______________________________________________________________________________
void TGWin32::ChangeWindowAttributes(Window_t id,
                                     SetWindowAttributes_t * attr)
{
   // Change window attributes.
   GdkWMDecoration deco;
   GdkColor color;
   GdkEventMask xevent_mask;
   UInt_t xevmask;
   Mask_t evmask;
   HWND w, flag;

   if (attr) {
      color.pixel = attr->fBackgroundPixel;
      color.red = GetRValue(attr->fBackgroundPixel);
      color.green = GetGValue(attr->fBackgroundPixel);
      color.blue = GetBValue(attr->fBackgroundPixel);
   }
   if (attr && (attr->fMask & kWAEventMask)) {
      evmask = (Mask_t) attr->fEventMask;
      MapEventMask(evmask, xevmask);
      gdk_window_set_events((GdkWindow *) id, (GdkEventMask) xevmask);
   }
   if (attr && (attr->fMask & kWABackPixel))
      gdk_window_set_background((GdkWindow *) id, &color);
//   if (attr && (attr->fMask & kWAOverrideRedirect))
//      gdk_window_set_override_redirect ((GdkWindow *) id, attr->fOverrideRedirect);
   if (attr && (attr->fMask & kWABackPixmap))
      gdk_window_set_back_pixmap((GdkWindow *) id,
                                 (GdkPixmap *) attr->fBackgroundPixmap, 0);
   if (attr && (attr->fMask & kWACursor))
      gdk_window_set_cursor((GdkWindow *) id, (GdkCursor *) attr->fCursor);
   if (attr && (attr->fMask & kWAColormap))
      gdk_window_set_colormap((GdkWindow *) id,
                              (GdkColormap *) attr->fColormap);
   if (attr && (attr->fMask & kWABorderWidth)) {
      if (attr->fBorderWidth > 0)
         gdk_window_set_decorations((GdkWindow *) id,
                                    (GdkWMDecoration) GDK_DECOR_BORDER);
   }
}

//______________________________________________________________________________
void TGWin32::ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                             UChar_t * data, Int_t len)
{
   // This function alters the property for the specified window and
   // causes the X server to generate a PropertyNotify event on that
   // window.
   gdk_property_change((GdkWindow *) id, (GdkAtom) property,
                       (GdkAtom) type, 8, GDK_PROP_MODE_REPLACE, data,
                       len);
}

//______________________________________________________________________________
void TGWin32::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1,
                       Int_t x2, Int_t y2)
{
   // Draw a line.

   gdk_draw_line((GdkDrawable *) id, (GdkGC *) gc, x1, y1, x2, y2);
}

//______________________________________________________________________________
void TGWin32::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.

   gdk_window_clear_area((GdkWindow *) id, x, y, w, h);
}

//______________________________________________________________________________
Bool_t TGWin32::CheckEvent(Window_t id, EGEventType type, Event_t & ev)
{
   // Check if there is for window "id" an event of type "type". If there
   // is fill in the event structure and return true. If no such event
   // return false.
   Event_t tev;
   GdkEvent xev;

   tev.fType = type;
   MapEvent(tev, xev, kTRUE);
   Bool_t r =
       gdk_check_typed_window_event((GdkWindow *) id, xev.type, &xev);
   if (r)
      MapEvent(ev, xev, kFALSE);
   return r ? kTRUE : kFALSE;

}

//______________________________________________________________________________
void TGWin32::SendEvent(Window_t id, Event_t * ev)
{
   // Send event ev to window id.
   if (!ev)
      return;

   GdkEvent xev;

   MapEvent(*ev, xev, kTRUE);
   gdk_event_put(&xev);
//   gdk_flush();

}

//______________________________________________________________________________
void TGWin32::WMDeleteNotify(Window_t id)
{
   // Tell WM to send message when window is closed via WM.
   Atom prop;
   prop = (Atom_t) gdk_atom_intern("WM_DELETE_WINDOW", FALSE);

   W32ChangeProperty((HWND) GDK_DRAWABLE_XID((GdkWindow *) id),
                     prop, XA_ATOM, 32, GDK_PROP_MODE_REPLACE,
                     (unsigned char *) &gWM_DELETE_WINDOW, 1);

}

//______________________________________________________________________________
void TGWin32::SetKeyAutoRepeat(Bool_t on)
{
   // Turn key auto repeat on or off.

   if (on)
      gdk_key_repeat_restore();
   else
      gdk_key_repeat_disable();
}

//______________________________________________________________________________
void TGWin32::GrabKey(Window_t id, Int_t keycode, UInt_t modifier,
                      Bool_t grab)
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.
   UInt_t xmod;
   GdkEventMask masque;

   MapModifierState(modifier, xmod);
   if (grab) {
      masque = gdk_window_get_events((GdkWindow *) id);
      masque = (GdkEventMask) (masque | (GdkEventMask) xmod);
      gdk_window_set_events((GdkWindow *) id, masque);
      gdk_keyboard_grab((GdkWindow *) id, 1, GDK_CURRENT_TIME);
   } else {
      masque = gdk_window_get_events((GdkWindow *) id);
      masque = (GdkEventMask) (masque & (GdkEventMask) xmod);
      gdk_window_set_events((GdkWindow *) id, masque);
      gdk_keyboard_ungrab(GDK_CURRENT_TIME);
   }
}

//______________________________________________________________________________
void TGWin32::GrabButton(Window_t id, EMouseButton button, UInt_t modifier,
                         UInt_t evmask, Window_t confine, Cursor_t cursor,
                         Bool_t grab)
{
   // Establish passive grab on a certain mouse button. That is, when a
   // certain mouse button is hit while certain modifier's (Shift, Control,
   // Meta, Alt) are active then the mouse will be grabed for window id.
   // When grab is false, ungrab the mouse button for this button and modifier.
   UInt_t xevmask;
   GdkEventMask masque;
   UInt_t xmod;

   MapModifierState(modifier, xmod);
   MapEventMask(evmask, xevmask);
   if (grab) {
      masque = gdk_window_get_events((GdkWindow *) id);
      masque = (GdkEventMask) (masque | (GdkEventMask) xevmask);
      gdk_window_set_events((GdkWindow *) id, masque);
//       gdk_pointer_grab((GdkWindow *) id, 1, (GdkEventMask)xevmask,
//           (GdkWindow *) confine,(GdkCursor *) cursor, 0L);
   } else {
      masque = gdk_window_get_events((GdkWindow *) id);
      masque = (GdkEventMask) (masque & (GdkEventMask) xevmask);
      gdk_window_set_events((GdkWindow *) id, masque);
//      gdk_pointer_ungrab(0L);
   }

}

//______________________________________________________________________________
void TGWin32::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                          Cursor_t cursor, Bool_t grab,
                          Bool_t owner_events)
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.

   UInt_t xevmask;
   MapEventMask(evmask, xevmask);
   if (grab) {
      gdk_pointer_grab((GdkWindow *) id, owner_events,
                       (GdkEventMask) xevmask, (GdkWindow *) confine,
                       (GdkCursor *) cursor, GDK_CURRENT_TIME);
   } else {
      gdk_pointer_ungrab(GDK_CURRENT_TIME);
   }
}

//______________________________________________________________________________
void TGWin32::SetWindowName(Window_t id, char *name)
{
   // Set window name.
   gdk_window_set_title((GdkWindow *) id, name);
}

//______________________________________________________________________________
void TGWin32::SetIconName(Window_t id, char *name)
{
   // Set window icon name.

   gdk_window_set_icon_name((GdkWindow *) id, name);
}

//______________________________________________________________________________
void TGWin32::SetIconPixmap(Window_t id, Pixmap_t pic)
{
   // Set pixmap the WM can use when the window is iconized.
   gdk_window_set_icon((GdkWindow *) id, NULL, (GdkPixmap *) pic,
                       (GdkPixmap *) pic);

}

#define safestrlen(s) ((s) ? strlen(s) : 0)

//______________________________________________________________________________
void TGWin32::SetClassHints(Window_t id, char *className,
                            char *resourceName)
{
   // Set the windows class and resource name.
   char *class_string;
   char *s;
   int len_nm, len_cl;
   GdkAtom type, prop;

   prop = gdk_atom_intern("WM_CLASS", FALSE);

   len_nm = safestrlen(resourceName);
   len_cl = safestrlen(className);
   if ((class_string = s =
        (char *) malloc((unsigned) (len_nm + len_cl + 2)))) {
      if (len_nm) {
         strcpy(s, resourceName);
         s += len_nm + 1;
      } else
         *s++ = '\0';
      if (len_cl)
         strcpy(s, className);
      else
         *s = '\0';
//        gdk_property_change ((GdkWindow *) id, prop, GDK_TARGET_STRING,
//                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 8, GDK_PROP_MODE_REPLACE, (unsigned char *) class_string,
//                  len_nm+len_cl+2);
      W32ChangeProperty((HWND) GDK_DRAWABLE_XID((GdkWindow *) id),
                        (Atom_t) XA_WM_CLASS, (Atom_t) XA_STRING, 8,
                        GDK_PROP_MODE_REPLACE,
                        (unsigned char *) class_string,
                        len_nm + len_cl + 2);
      free(class_string);
   }
}

//______________________________________________________________________________
void TGWin32::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs,
                          UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).
   gdk_window_set_decorations((GdkWindow *) id, (GdkWMDecoration) value);
   gdk_window_set_functions((GdkWindow *) id, (GdkWMFunction) funcs);
}

//______________________________________________________________________________
void TGWin32::SetWMPosition(Window_t id, Int_t x, Int_t y)
{

   gdk_window_move((GdkWindow *) id, x, y);
}

//______________________________________________________________________________
void TGWin32::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{

   gdk_window_resize((GdkWindow *) id, w, h);
}

//______________________________________________________________________________
void TGWin32::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                             UInt_t wmax, UInt_t hmax,
                             UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.

   GdkGeometry hints;
   GdkWindowHints flags;

   flags =
       (GdkWindowHints) (GDK_HINT_MIN_SIZE | GDK_HINT_MAX_SIZE |
                         GDK_HINT_RESIZE_INC);
   hints.min_width = (Int_t) wmin;
   hints.max_width = (Int_t) wmax;
   hints.min_height = (Int_t) hmin;
   hints.max_height = (Int_t) hmax;
   hints.width_inc = (Int_t) winc;
   hints.height_inc = (Int_t) hinc;

   gdk_window_set_geometry_hints((GdkWindow *) id, &hints, flags);
}

//______________________________________________________________________________
void TGWin32::SetWMState(Window_t id, EInitialState state)
{
   // Set the initial state of the window. Either kNormalState or kIconicState.
#if 0
   XWMHints hints;
   Int_t xstate = NormalState;

   if (state == kNormalState)
      xstate = NormalState;
   if (state == kIconicState)
      xstate = IconicState;

   hints.flags = StateHint;
   hints.initial_state = xstate;

   XSetWMHints((GdkWindow *) id, &hints);
#endif
}

//______________________________________________________________________________
void TGWin32::SetWMTransientHint(Window_t id, Window_t main_id)
{
   // Tell window manager that window is a transient window of gdk_parent_root.

   gdk_window_set_transient_for((GdkWindow *) id, (GdkWindow *) main_id);
}

//______________________________________________________________________________
void TGWin32::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                         const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y).
   GdkGCValues values;
   gdk_gc_get_values((GdkGC *) gc, &values);

   gdk_draw_text((GdkDrawable *) id, (GdkFont *) values.font, (GdkGC *) gc,
                 x, y, (char *) s, len);
}

//______________________________________________________________________________
Int_t TGWin32::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font.

   return gdk_text_width((GdkFont *) font, (char *) s, len);
}

//______________________________________________________________________________
void TGWin32::GetFontProperties(FontStruct_t font, Int_t & max_ascent,
                                Int_t & max_descent)
{
   // Return some font properties.

   GdkFont *f = (GdkFont *) font;

   max_ascent = f->ascent;
   max_descent = f->descent;
}

//______________________________________________________________________________
void TGWin32::GetGCValues(GContext_t gc, GCValues_t & gval)
{
   // Get current values from graphics context gc. Which values of the
   // context to get is encoded in the GCValues::fMask member.

   GdkGCValues xgval;
   ULong_t xmask;

   MapGCValues(gval, xmask, xgval, kTRUE);

   gdk_gc_get_values((GdkGC *) gc, &xgval);

   MapGCValues(gval, xmask, xgval, kFALSE);
}

//______________________________________________________________________________
FontStruct_t TGWin32::GetFontStruct(FontH_t fh)
{
   // Retrieve associated font structure once we have the font handle.
   // Free returned FontStruct_t using FreeFontStruct().
   GdkFont *fs;

   fs = gdk_font_ref((GdkFont *) fh);

   return (FontStruct_t) fs;
}

//______________________________________________________________________________
void TGWin32::FreeFontStruct(FontStruct_t fs)
{
   // Free font structure returned by GetFontStruct().
   gdk_font_unref((GdkFont *) fs);
}

//______________________________________________________________________________
void TGWin32::ClearWindow(Window_t id)
{
   // Clear window.

   gdk_window_clear((GdkWindow *) id);
}

//______________________________________________________________________________
Int_t TGWin32::KeysymToKeycode(UInt_t keysym)
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).
   UInt_t xkeysym;

   MapKeySym(keysym, xkeysym);
   xkeysym = gdk_keyval_from_name((const char *) &xkeysym);
   return xkeysym;
   return 0;
}

//______________________________________________________________________________
void TGWin32::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                            UInt_t w, UInt_t h)
{
   // Draw a filled rectangle. Filling is done according to the gc.

   gdk_draw_rectangle((GdkDrawable *) id, (GdkGC *) gc, 1, x, y, w, h);
}

//______________________________________________________________________________
void TGWin32::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                            UInt_t w, UInt_t h)
{
   // Draw a rectangle outline.

   gdk_draw_rectangle((GdkDrawable *) id, (GdkGC *) gc, 0, x, y, w, h);
}

//______________________________________________________________________________
void TGWin32::DrawSegments(Drawable_t id, GContext_t gc, Segment_t * seg,
                           Int_t nseg)
{
   // Draws multiple line segments. Each line is specified by a pair of points.

   gdk_draw_segments((GdkDrawable *) id, (GdkGC *) gc, (GdkSegment *) seg,
                     nseg);
}

//______________________________________________________________________________
void TGWin32::SelectInput(Window_t id, UInt_t evmask)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.

   UInt_t xevmask;
   GdkEventMask masque;
   GdkEventMask tmp_masque;

   MapEventMask(evmask, xevmask, kTRUE);

   tmp_masque = gdk_window_get_events((GdkWindow *) id);
   masque = (GdkEventMask) (tmp_masque | (GdkEventMask) xevmask);
   gdk_window_set_events((GdkWindow *) id, masque);
}

//______________________________________________________________________________
Window_t TGWin32::GetInputFocus()
{
   // Returns the window id of the window having the input focus.

   HWND focuswindow;

   focuswindow = GetFocus();
   return (Window_t) gdk_xid_table_lookup(focuswindow);
}

//______________________________________________________________________________
void TGWin32::SetInputFocus(Window_t id)
{
   // Set keyboard input focus to window id.

   SetFocus((HWND) GDK_DRAWABLE_XID((GdkWindow *) id));
}

//______________________________________________________________________________
Window_t TGWin32::GetPrimarySelectionOwner()
{
   // Returns the window id of the current owner of the primary selection.
   // That is the window in which, for example some text is selected.
   return (Window_t)
       gdk_selection_owner_get(gdk_atom_intern
                               ("GDK_SELECTION_PRIMARY", 0));
}

//______________________________________________________________________________
void TGWin32::SetPrimarySelectionOwner(Window_t id)
{
   // Makes the window id the current owner of the primary selection.
   // That is the window in which, for example some text is selected.
   gdk_selection_owner_set((GdkWindow *) id, clipboard_atom,
                           GDK_CURRENT_TIME, 0);
}

//______________________________________________________________________________
void TGWin32::ConvertPrimarySelection(Window_t id, Atom_t clipboard,
                                      Time_t when)
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
   gdk_selection_convert((GdkWindow *) id, clipboard,
                         gdk_atom_intern("GDK_TARGET_STRING", 0), when);
}

Int_t _lookup_string(Event_t * event, char *buf, Int_t buflen)
{
   int i;
   int n = event->fUser[1];
   if (n > 0) {
      for (i = 0; i < n; i++)
         buf[i] = event->fUser[2 + i];
      buf[n] = 0;
   } else
      buf[0] = 0;
   if (event->fCode <= 0x20) {
      strncpy(buf, keyCodeToString[event->fCode], buflen - 1);
   }
   return n;
}

//______________________________________________________________________________
void TGWin32::LookupString(Event_t * event, char *buf, Int_t buflen,
                           UInt_t & keysym)
{
   // Convert the keycode from the event structure to a key symbol (according
   // to the modifiers specified in the event structure and the current
   // keyboard mapping). In buf a null terminated ASCII string is returned
   // representing the string that is currently mapped to the key code.

   KeySym xkeysym;
   _lookup_string(event, buf, buflen);
   UInt_t ks, xks = (UInt_t) event->fCode;
   MapKeySym(ks, xks, kFALSE);
   keysym = (Int_t) ks;
}

//______________________________________________________________________________
void TGWin32::MapKeySym(UInt_t & keysym, UInt_t & xkeysym, Bool_t tox)
{
   // Map to and from X key symbols. Keysym are the values returned by
   // XLookUpString.

   if (tox) {
      xkeysym = GDK_VoidSymbol;
      if (keysym < 127) {
         xkeysym = keysym;
      } else if (keysym >= kKey_F1 && keysym <= kKey_F35) {
         xkeysym = GDK_F1 + (keysym - (UInt_t) kKey_F1);	// function keys
      } else {
         for (int i = 0; gKeyMap[i].fKeySym; i++) {	// any other keys
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
      } else if (xkeysym >= GDK_F1 && xkeysym <= GDK_F35) {
         keysym = kKey_F1 + (xkeysym - GDK_F1);	// function keys
      } else if (xkeysym >= GDK_KP_0 && xkeysym <= GDK_KP_9) {
         keysym = kKey_0 + (xkeysym - GDK_KP_0);	// numeric keypad keys
      } else {
         for (int i = 0; gKeyMap[i].fXKeySym; i++) {	// any other keys
            if (xkeysym == gKeyMap[i].fXKeySym) {
               keysym = (UInt_t) gKeyMap[i].fKeySym;
               break;
            }
         }
      }
   }

}

//______________________________________________________________________________
void TGWin32::GetPasteBuffer(Window_t id, Atom_t atom, TString & text,
                             Int_t & nchar, Bool_t del)
{
   // Get contents of paste buffer atom into string. If del is true delete
   // the paste buffer afterwards.

   char *data;
   int nread, actual_format;

   nread = gdk_selection_property_get((GdkWindow *) id,
                                      (unsigned char **) &data,
                                      (GdkAtom *) & atom, &actual_format);
//    printf("TGWin32::GetPasteBuffer : %s\n",data);
   if ((nread == 0) || (data == NULL)) {
      nchar = 0;
      return;
   }
   text.Insert(0, (const char *) data);
   nchar = 1;                   //strlen(data);
   g_free(data);
   if (del)
      gdk_property_delete((GdkWindow *) id,
                          gdk_atom_intern("GDK_SELECTION", FALSE));

}

//______________________________________________________________________________
void TGWin32::TranslateCoordinates(Window_t src, Window_t dest,
                                   Int_t src_x, Int_t src_y,
                                   Int_t & dest_x, Int_t & dest_y,
                                   Window_t & child)
{
   // TranslateCoordinates translates coordinates from the frame of
   // reference of one window to another. If the point is contained
   // in a mapped child of the destination, the id of that child is
   // returned as well.
   HWND sw, dw, ch = NULL;
   POINT point;
   sw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) src);
   dw = (HWND) GDK_DRAWABLE_XID((GdkWindow *) dest);
   point.x = src_x;
   point.y = src_y;
   MapWindowPoints(sw,          // handle of window to be mapped from
                   dw,          // handle to window to be mapped to
                   &point,      // pointer to array with points to map
                   1);          // number of structures in array
   ch = ChildWindowFromPoint(dw, point);
   child = (Window_t) gdk_xid_table_lookup(ch);
   if (child == src)
      child = (Window_t) 0;
   dest_x = point.x;
   dest_y = point.y;
}

//______________________________________________________________________________
void TGWin32::GetWindowSize(Drawable_t id, Int_t & x, Int_t & y,
                            UInt_t & w, UInt_t & h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).

   Int_t ddum;

   gdk_window_get_geometry((GdkDrawable *) id, &x, &y, (int *) &w,
                           (int *) &h, &ddum);
}

//______________________________________________________________________________
void TGWin32::FillPolygon(Window_t id, GContext_t gc, Point_t * points,
                          Int_t npnt)
{
   // FillPolygon fills the region closed by the specified path.
   // The path is closed automatically if the last point in the list does
   // not coincide with the first point. All point coordinates are
   // treated as relative to the origin. For every pair of points
   // inside the polygon, the line segment connecting them does not
   // intersect the path.

   gdk_draw_polygon((GdkWindow *) id, (GdkGC *) gc, 1, (GdkPoint *) points,
                    npnt);
}

//______________________________________________________________________________
void TGWin32::QueryPointer(Window_t id, Window_t & rootw,
                           Window_t & childw, Int_t & root_x,
                           Int_t & root_y, Int_t & win_x, Int_t & win_y,
                           UInt_t & mask)
{
   // Returns the root window the pointer is logically on and the pointer
   // coordinates relative to the root window's origin.
   // The pointer coordinates returned to win_x and win_y are relative to
   // the origin of the specified window. In this case, QueryPointer returns
   // the child that contains the pointer, if any, or else kNone to
   // childw. QueryPointer returns the current logical state of the
   // keyboard buttons and the modifier keys in mask.
   POINT mousePt, sPt, currPt;
   HWND chw, window;
   UInt_t ev_mask = 0;

   window = (HWND) GDK_DRAWABLE_XID((GdkWindow *) id);
   rootw = (Window_t) GDK_ROOT_PARENT();
   GetCursorPos(&currPt);
   chw = ChildWindowFromPoint(window, currPt);
   ClientToScreen(window, &mousePt);
   root_x = mousePt.x;
   root_y = mousePt.y;
   sPt.x = mousePt.x;
   sPt.y = mousePt.y;
   ScreenToClient(window, &sPt);
   win_x = sPt.x;
   win_y = sPt.y;
   mask = 0L;
   childw = (Window_t) gdk_xid_table_lookup(chw);
   if (childw)
      ev_mask = (UInt_t) gdk_window_get_events((GdkWindow *) childw);
   MapEventMask(mask, ev_mask, kFALSE);

}

//______________________________________________________________________________
void TGWin32::SetForeground(GContext_t gc, ULong_t foreground)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).
   GdkColor fore;
   fore.pixel = foreground;
   fore.red = GetRValue(foreground);
   fore.green = GetGValue(foreground);
   fore.blue = GetBValue(foreground);

   gdk_gc_set_foreground((GdkGC *) gc, &fore);
}

//______________________________________________________________________________
void TGWin32::SetClipRectangles(GContext_t gc, Int_t x, Int_t y,
                                Rectangle_t * recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.

   Int_t i;
   GdkRectangle *grects = new GdkRectangle[n];
   for (i = 0; i < n; i++) {
      grects[i].x = recs[i].fX;
      grects[i].y = recs[i].fY;
      grects[i].width = recs[i].fWidth;
      grects[i].height = recs[i].fHeight;
   }
   gdk_gc_set_clip_rectangle((GdkGC *) gc, grects);
}

//______________________________________________________________________________
void TGWin32::Update(Int_t mode)
{
   // Flush (mode = 0, default) or synchronize (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.

   if (mode == 0)
      gdk_flush();
   if (mode == 1) {
      while (gdk_events_pending())
         gdk_flush();
   }
}

//______________________________________________________________________________
Region_t TGWin32::CreateRegion()
{
   // Create a new empty region.

   return (Region_t) gdk_region_new();
}

//______________________________________________________________________________
void TGWin32::DestroyRegion(Region_t reg)
{
   // Destroy region.

   gdk_region_destroy((GdkRegion *) reg);
}

//______________________________________________________________________________
void TGWin32::UnionRectWithRegion(Rectangle_t * rect, Region_t src,
                                  Region_t dest)
{
   // Union of rectangle with a region.

   GdkRectangle r;
   r.x = rect->fX;
   r.y = rect->fY;
   r.width = rect->fWidth;
   r.height = rect->fHeight;
   dest = (Region_t) gdk_region_union_with_rect((GdkRegion *) src, &r);
}

//______________________________________________________________________________
Region_t TGWin32::PolygonRegion(Point_t * points, Int_t np, Bool_t winding)
{
   // Create region for the polygon defined by the points array.
   // If winding is true use WindingRule else EvenOddRule as fill rule.

   int i;
   GdkPoint *xy = new GdkPoint[np];

   for (i = 0; i < np; i++) {
      xy[i].x = points[i].fX;
      xy[i].y = points[i].fY;
   }
   return (Region_t) gdk_region_polygon(xy, np,
                                        winding ? GDK_WINDING_RULE :
                                        GDK_EVEN_ODD_RULE);
}

//______________________________________________________________________________
void TGWin32::UnionRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the union of rega and regb and return result region.
   // The output region may be the same result region.

   result =
       (Region_t) gdk_regions_union((GdkRegion *) rega,
                                    (GdkRegion *) regb);
}

//______________________________________________________________________________
void TGWin32::IntersectRegion(Region_t rega, Region_t regb,
                              Region_t result)
{
   // Compute the intersection of rega and regb and return result region.
   // The output region may be the same as the result region.

   result =
       (Region_t) gdk_regions_intersect((GdkRegion *) rega,
                                        (GdkRegion *) regb);
}

//______________________________________________________________________________
void TGWin32::SubtractRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Subtract rega from regb.

   result =
       (Region_t) gdk_regions_subtract((GdkRegion *) rega,
                                       (GdkRegion *) regb);
}

//______________________________________________________________________________
void TGWin32::XorRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Calculate the difference between the union and intersection of
   // two regions.

   result =
       (Region_t) gdk_regions_xor((GdkRegion *) rega, (GdkRegion *) regb);
}

//______________________________________________________________________________
Bool_t TGWin32::EmptyRegion(Region_t reg)
{
   // Return true if the region is empty.

   return (Bool_t) gdk_region_empty((GdkRegion *) reg);
}

//______________________________________________________________________________
Bool_t TGWin32::PointInRegion(Int_t x, Int_t y, Region_t reg)
{
   // Returns true if the point x,y is in the region.

   return (Bool_t) gdk_region_point_in((GdkRegion *) reg, x, y);
}

//______________________________________________________________________________
Bool_t TGWin32::EqualRegion(Region_t rega, Region_t regb)
{
   // Returns true if two regions are equal.

   return (Bool_t) gdk_region_equal((GdkRegion *) rega,
                                    (GdkRegion *) regb);
}

//______________________________________________________________________________
void TGWin32::GetRegionBox(Region_t reg, Rectangle_t * rect)
{
   // Return smallest enclosing rectangle.

   GdkRectangle r;
   gdk_region_get_clipbox((GdkRegion *) reg, &r);
   rect->fX = r.x;
   rect->fY = r.y;
   rect->fWidth = r.width;
   rect->fHeight = r.height;
}
