// @(#)root/win32gdk:$Name:  $:$Id: GWin32Gui.cxx,v 1.10 2002/10/25 17:38:00 rdm Exp $
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

#ifndef ROOT_GdkConstants
#include "GdkConstants.h"
#endif

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
   EnterCriticalSection(flpCriticalSection);
   // Map window on screen.
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SHOW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::MapSubwindows(Window_t id)
{
   EnterCriticalSection(flpCriticalSection);
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_MAP_SUBWINDOWS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::MapRaised(Window_t id)
{
   EnterCriticalSection(flpCriticalSection);
   // Map window on screen and put on top of all windows.
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SHOW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RAISE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::UnmapWindow(Window_t id)
{
   // Unmap window from screen.
   EnterCriticalSection(flpCriticalSection);
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_HIDE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DestroyWindow(Window_t id)
{
   // Destroy window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_DESTROY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::RaiseWindow(Window_t id)
{
   // Put window on top of window stack.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RAISE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::LowerWindow(Window_t id)
{
   // Lower window so it lays below all its siblings.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_LOWER, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::MoveWindow(Window_t id, Int_t x, Int_t y)
{
   // Move a window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.x = x;
   fThreadP.y = y;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_MOVE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::MoveResizeWindow(Window_t id, Int_t x, Int_t y, UInt_t w,
                               UInt_t h)
{
   // Move and resize a window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_MOVE_RESIZE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ResizeWindow(Window_t id, UInt_t w, UInt_t h)
{
   // Resize the window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RESIZE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::IconifyWindow(Window_t id)
{
   // Iconify the window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_LOWER, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWindowBackground(Window_t id, ULong_t color)
{
   // Set the window background color.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.color.pixel = color;
   fThreadP.color.red = GetRValue(color);
   fThreadP.color.green = GetGValue(color);
   fThreadP.color.blue = GetBValue(color);
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACKGROUND, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWindowBackgroundPixmap(Window_t id, Pixmap_t pxm)
{
   // Set pixmap as window background.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *)pxm;
   fThreadP.iParam = 0;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACK_PIXMAP, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Window_t TGWin32::CreateWindow(Window_t parent, Int_t x, Int_t y,
                               UInt_t w, UInt_t h, UInt_t border,
                               Int_t depth, UInt_t clss,
                               void *visual, SetWindowAttributes_t * attr,
                               UInt_t wtype)
{
   // Return handle to newly created gdk window.
   EnterCriticalSection(flpCriticalSection);

   GdkWMDecoration deco;
   GdkWindow *newWin;
   GdkColor background_color;
   ULong_t xmask = 0;
   UInt_t xevmask;

   if (attr) {
      MapSetWindowAttributes(attr, xmask, fThreadP.xattr);
      fThreadP.xattr.window_type = GDK_WINDOW_CHILD;
      if (wtype & kTransientFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_DIALOG;
      }
      if (wtype & kMainFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_TOPLEVEL;
      }
      if (wtype & kTempFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_TEMP;
      }
      fThreadP.Drawable = (GdkDrawable *) parent;
      fThreadP.lParam = xmask;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_NEW, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      newWin = (GdkWindow *) fThreadP.pRet;
   } else {
      fThreadP.xattr.width = w;
      fThreadP.xattr.height = h;
      fThreadP.xattr.wclass = GDK_INPUT_OUTPUT;
      fThreadP.xattr.event_mask = 0L;    //GDK_ALL_EVENTS_MASK;
      fThreadP.xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK |
          GDK_PROPERTY_CHANGE_MASK;
//                          GDK_ENTER_NOTIFY_MASK | GDK_LEAVE_NOTIFY_MASK;
      if (x >= 0)
         fThreadP.xattr.x = x;
      else
         fThreadP.xattr.x = -1.0 * x;
      if (y >= 0)
         fThreadP.xattr.y = y;
      else
         fThreadP.xattr.y = -1.0 * y;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_CMAP_GET_SYSTEM, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      fThreadP.xattr.colormap = (GdkColormap *)fThreadP.pRet;
      fThreadP.xattr.cursor = NULL;
      fThreadP.xattr.override_redirect = TRUE;
      if ((fThreadP.xattr.y > 0) && (fThreadP.xattr.x > 0))
         xmask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP |
             GDK_WA_WMCLASS | GDK_WA_NOREDIR;
      else
         xmask = GDK_WA_COLORMAP | GDK_WA_WMCLASS | GDK_WA_NOREDIR;
      if (visual != NULL) {
         fThreadP.xattr.visual = (GdkVisual *) visual;
         xmask |= GDK_WA_VISUAL;
      } else {
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_VISUAL_GET_SYSTEM, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         fThreadP.xattr.visual = (GdkVisual *)fThreadP.pRet;
         xmask |= GDK_WA_VISUAL;
      }
      fThreadP.xattr.window_type = GDK_WINDOW_CHILD;
      if (wtype & kTransientFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_DIALOG;
      }
      if (wtype & kMainFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_TOPLEVEL;
      }
      if (wtype & kTempFrame) {
         fThreadP.xattr.window_type = GDK_WINDOW_TEMP;
      }
      fThreadP.Drawable = (GdkDrawable *) parent;
      fThreadP.lParam = xmask;
      fThreadP.pRet = NULL;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_NEW, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      newWin = (GdkWindow *)fThreadP.pRet;

      fThreadP.Drawable = (GdkDrawable *) newWin;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_EVENTS, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (border > 0) {
      fThreadP.Drawable = (GdkDrawable *) newWin;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_DECOR, GDK_DECOR_BORDER, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (attr) {
      if ((attr->fMask & kWABackPixmap)) {
         if (attr->fBackgroundPixmap == kNone) {
            fThreadP.Drawable = (GdkDrawable *) newWin;
            fThreadP.pParam = (void *) GDK_NONE;
            fThreadP.iParam = 0;
            PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACK_PIXMAP, 0, 0L);  
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         } else if (attr->fBackgroundPixmap == kParentRelative) {
            fThreadP.Drawable = (GdkDrawable *) newWin;
            fThreadP.pParam = (void *) GDK_NONE;
            fThreadP.iParam = 1;
            PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACK_PIXMAP, 0, 0L);  
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         } else {
            fThreadP.Drawable = (GdkDrawable *) newWin;
            fThreadP.pParam = (void *) attr->fBackgroundPixmap;
            fThreadP.iParam = 0;
            PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACK_PIXMAP, 0, 0L);  
            WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         }
      }
      if ((attr->fMask & kWABackPixel)) {
         fThreadP.Drawable = (GdkDrawable *) newWin;
         fThreadP.color.pixel = attr->fBackgroundPixel;
         fThreadP.color.red = GetRValue(attr->fBackgroundPixel);
         fThreadP.color.green = GetGValue(attr->fBackgroundPixel);
         fThreadP.color.blue = GetBValue(attr->fBackgroundPixel);
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACKGROUND, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   GdkWindowAttr xattr;
   Bool_t isViewable, isVisible;

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   attr.fX = fThreadP.x;
   attr.fY = fThreadP.y;
   attr.fWidth = fThreadP.w;
   attr.fHeight = fThreadP.h;
   attr.fDepth = fThreadP.iRet;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_ROOT_PARENT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   
   attr.fRoot = (Window_t) fThreadP.pRet;
   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GET_COLORMAP, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   attr.fColormap = (Colormap_t) fThreadP.pRet;
   attr.fBorderWidth = 0;
   fThreadP.pRet = NULL;
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GET_VISUAL, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   attr.fVisual = fThreadP.pRet;
   attr.fClass = kInputOutput;
   attr.fBackingStore = kNotUseful;
   attr.fSaveUnder = kFALSE;
   attr.fMapInstalled = kTRUE;
   attr.fOverrideRedirect = kFALSE;   // boolean value for override-redirect
   fThreadP.iRet = 0;
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_IS_VISIBLE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   isVisible = fThreadP.iRet;
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_IS_VIEWABLE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   isViewable = fThreadP.iRet;
   if (!isVisible)
      attr.fMapState = kIsUnmapped;
   else if (!isViewable)
      attr.fMapState = kIsUnviewable;
   else
      attr.fMapState = kIsViewable;
   LeaveCriticalSection(flpCriticalSection);
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

    if (fIDThread) {
        PostThreadMessage(fIDThread, WIN32_GDK_EXIT, 0, 0L);  
        WaitForSingleObject(fThreadP.hThrSem, INFINITE);
        CloseHandle(fThreadP.hThrSem);
        CloseHandle(hGDKThread);
    }

    DeleteCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Display_t TGWin32::GetDisplay() const
{
   return 0;
}

//______________________________________________________________________________
Int_t TGWin32::GetDepth() const
{
   // Get maximum number of planes.
   EnterCriticalSection(flpCriticalSection);

   PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return fThreadP.iRet;
}

//______________________________________________________________________________
Atom_t TGWin32::InternAtom(const char *atom_name, Bool_t only_if_exist)
{
   // Return atom handle for atom_name. If it does not exist
   // create it if only_if_exist is false. Atoms are used to communicate
   // between different programs (i.e. window manager) via the X server.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.iParam = only_if_exist;
   sprintf(fThreadP.sParam,"%s",atom_name);
   PostThreadMessage(fIDThread, WIN32_GDK_ATOM, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkAtom a = (GdkAtom)fThreadP.ulRet;
   LeaveCriticalSection(flpCriticalSection);
   if (a == None)
      return kNone;
   return (Atom_t) a;
}

//______________________________________________________________________________
Window_t TGWin32::GetDefaultRootWindow() const
{
   // Return handle to the default root window created when calling
   // XOpenDisplay().
   EnterCriticalSection(flpCriticalSection);

//   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_ROOT_PARENT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   
   LeaveCriticalSection(flpCriticalSection);
   return (Window_t) fThreadP.pRet;
}

//______________________________________________________________________________
Window_t TGWin32::GetParent(Window_t id) const
{
   // Return the parent of the window.
   EnterCriticalSection(flpCriticalSection);

//   fThreadP.lParam = id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GET_PARENT, 0, id);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   
   LeaveCriticalSection(flpCriticalSection);
   return (Window_t) fThreadP.pRet;
//   return (Window_t) NULL;
}


//______________________________________________________________________________
FontStruct_t TGWin32::LoadQueryFont(const char *font_name)
{
   // Load font and query font. If font is not found 0 is returned,
   // otherwise an opaque pointer to the FontStruct_t.
   // Free the loaded font using DeleteFont().
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   sprintf(fThreadP.sParam,"%s",font_name);
   PostThreadMessage(fIDThread, WIN32_GDK_FONT_LOAD, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkFont *fs = (GdkFont *)fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
   return (FontStruct_t) fs;
}

//______________________________________________________________________________
FontH_t TGWin32::GetFontHandle(FontStruct_t fs)
{
   // Return handle to font described by font structure.
   EnterCriticalSection(flpCriticalSection);

   if (fs) {
      fThreadP.pRet = NULL;
      fThreadP.pParam = (void *) fs;
      PostThreadMessage(fIDThread, WIN32_GDK_FONT_REF, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      GdkFont *fss = (GdkFont *)fThreadP.pRet;
      LeaveCriticalSection(flpCriticalSection);
      return (FontH_t) fss;
   }
   LeaveCriticalSection(flpCriticalSection);
   return 0;
}

//______________________________________________________________________________
void TGWin32::DeleteFont(FontStruct_t fs)
{
   // Explicitely delete font structure obtained with LoadQueryFont().
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) fs;
   PostThreadMessage(fIDThread, WIN32_GDK_FONT_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
GContext_t TGWin32::CreateGC(Drawable_t id, GCValues_t * gval)
{
   // Create a graphics context using the values set in gval (but only for
   // those entries that are in the mask).
   EnterCriticalSection(flpCriticalSection);

   GdkGCValues xgval;
   ULong_t xmask = 0;

   if (gval)
      MapGCValues(*gval, xmask, fThreadP.gcvals, kTRUE);
   fThreadP.gcvals.subwindow_mode = GDK_CLIP_BY_CHILDREN;	// GDK_INCLUDE_INFERIORS;

   fThreadP.pRet = NULL;
   fThreadP.Drawable = NULL;
   fThreadP.iParam = xmask;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_NEW_WITH_VAL, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkGC *gc = (GdkGC *)fThreadP.pRet;

   LeaveCriticalSection(flpCriticalSection);
   return (GContext_t) gc;
}

//______________________________________________________________________________
void TGWin32::ChangeGC(GContext_t gc, GCValues_t * gval)
{
   // Change entries in an existing graphics context, gc, by values from gval.
   EnterCriticalSection(flpCriticalSection);

   GdkGCValues xgval;
   ULong_t xmask = 0;
   Mask_t mask = 0;

   if (gval) {
      mask = gval->fMask;
      MapGCValues(*gval, xmask, xgval, kTRUE);
   }
   if (mask & kGCForeground) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.color.pixel = xgval.foreground.pixel;
      fThreadP.color.red   = xgval.foreground.red;
      fThreadP.color.green = xgval.foreground.green;
      fThreadP.color.blue  = xgval.foreground.blue;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCBackground) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.color.pixel = xgval.foreground.pixel;
      fThreadP.color.red   = xgval.foreground.red;
      fThreadP.color.green = xgval.foreground.green;
      fThreadP.color.blue  = xgval.foreground.blue;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_BACKGROUND, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCFont) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.pParam = (void *) xgval.font;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FONT, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCFunction) {
      fThreadP.GC = (GdkGC *) gc;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FUNCTION, xgval.function, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCFillStyle) {
      fThreadP.GC = (GdkGC *) gc;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FILL, xgval.fill, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCTile) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.pParam = (void *) xgval.tile;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_TILE, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCStipple) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.pParam = (void *) xgval.stipple;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_STIPPLE, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if ((mask & kGCTileStipXOrigin) || (mask & kGCTileStipYOrigin)) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.x = xgval.ts_x_origin;
      fThreadP.y = xgval.ts_y_origin;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_TS_ORIGIN, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if ((mask & kGCClipXOrigin) || (mask & kGCClipYOrigin)) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.x = xgval.clip_x_origin;
      fThreadP.y = xgval.clip_y_origin;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_ORIGIN, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCClipMask) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.pParam = (void *) xgval.clip_mask;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_MASK, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCGraphicsExposures) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.iParam = xgval.graphics_exposures;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_EXPOSURES, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if ((mask & kGCLineWidth) || (mask & kGCLineStyle) ||
       (mask & kGCCapStyle) || (mask & kGCJoinStyle)) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.w = xgval.line_width;
      fThreadP.iParam =  xgval.line_style;
      fThreadP.iParam1 = xgval.cap_style;
      fThreadP.iParam2 = xgval.join_style;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_LINE_ATTR, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (mask & kGCSubwindowMode) {
      fThreadP.GC = (GdkGC *) gc;
      fThreadP.iParam = xgval.subwindow_mode;
      PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_SUBWINDOW, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::CopyGC(GContext_t org, GContext_t dest, Mask_t mask)
{
   // Copies graphics context from org to dest. Only the values specified
   // in mask are copied. Both org and dest must exist.
   EnterCriticalSection(flpCriticalSection);

   GCValues_t gval;
   GdkGCValues xgval;
   ULong_t xmask;

   if (!mask) {
      // in this case copy all fields
      mask = (Mask_t) - 1;
   }

   gval.fMask = mask;           // only set fMask used to convert to xmask
   MapGCValues(gval, xmask, xgval, kTRUE);

   fThreadP.GC = (GdkGC *) dest;
   fThreadP.pParam = (void *) org;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_COPY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DeleteGC(GContext_t gc)
{
   // Explicitely delete a graphics context.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.GC = (GdkGC *) gc;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *) curid;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreatePixmap(Drawable_t id, UInt_t w, UInt_t h)
{
   // Creates a pixmap of the width and height you specified
   // and returns a pixmap ID that identifies it.
   EnterCriticalSection(flpCriticalSection);

   Int_t depth;
   PostThreadMessage(fIDThread, WIN32_GDK_GET_DEPTH, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   depth = fThreadP.iRet;
   
   fThreadP.pRet = NULL;
   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.iParam = depth;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_NEW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Pixmap_t) fThreadP.pRet;
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreatePixmap(Drawable_t id, const char *bitmap,
                               UInt_t width, UInt_t height,
                               ULong_t forecolor, ULong_t backcolor,
                               Int_t depth)
{
   // Create a pixmap from bitmap data. Ones will get foreground color and
   // zeroes background color.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *) bitmap;
   fThreadP.w = width;
   fThreadP.h = height;
   fThreadP.iParam = depth;
   fThreadP.lParam = forecolor;
   fThreadP.lParam1 = backcolor;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_CREATE_FROM_DATA, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Pixmap_t) fThreadP.pRet;
}

//______________________________________________________________________________
Pixmap_t TGWin32::CreateBitmap(Drawable_t id, const char *bitmap,
                               UInt_t width, UInt_t height)
{
   // Create a bitmap (i.e. pixmap with depth 1) from the bitmap data.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (char *) bitmap;
   fThreadP.w = width;
   fThreadP.h = height;
   PostThreadMessage(fIDThread, WIN32_GDK_BMP_CREATE_FROM_DATA, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Pixmap_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::DeletePixmap(Pixmap_t pmap)
{
   // Explicitely delete pixmap resource.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) pmap;
   PostThreadMessage(fIDThread, WIN32_GDK_PIX_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   GdkBitmap *gdk_pixmap_mask;
   fThreadP.Drawable = (GdkDrawable *) id;
   sprintf(fThreadP.sParam,"%s",filename);
   fThreadP.pRet = NULL;
   fThreadP.pParam1 = 0;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_CREATE_FROM_XPM, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   pict = (Pixmap_t) fThreadP.pRet;
   
   pict_mask = (Pixmap_t) fThreadP.pParam;

   fThreadP.Drawable = (GdkDrawable *) pict;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAWABLE_GET_SIZE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   attr.fWidth  = fThreadP.w;
   attr.fHeight = fThreadP.h;

   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   GdkBitmap *gdk_pixmap_mask;

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *) data;
   fThreadP.pParam1 = 0;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_CREATE_FROM_XPM_D, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   pict = (Pixmap_t) fThreadP.pRet;
   
   pict_mask = (Pixmap_t) fThreadP.pParam;

   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);
   char **rdata;

   fThreadP.Drawable = (GdkDrawable *) NULL;
   sprintf(fThreadP.sParam,"%s",filename);
   fThreadP.pParam1 = 0;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_PIXMAP_CREATE_FROM_XPM, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   rdata = (char **)fThreadP.pRet;
   
   ret_data = &rdata;
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   int i;
   for (i = 0; i < n; i++)
      fThreadP.dashes[i] = (gint8) dash_list[i];
   for (i = n; i < 32; i++)
      fThreadP.dashes[i] = (gint8) 0;

   fThreadP.GC = (GdkGC *) gc;
   fThreadP.iParam = offset;
   fThreadP.iParam2 = n;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_DASHES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   GdkColor xc;

   sprintf(fThreadP.sParam,"%s",cname);
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_PARSE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if (fThreadP.iRet) {
      color.fPixel = 
          fThreadP.color.pixel = 
          RGB(fThreadP.color.red, fThreadP.color.green, fThreadP.color.blue);
      color.fRed = fThreadP.color.red;
      color.fGreen = fThreadP.color.green;
      color.fBlue = fThreadP.color.blue;
      LeaveCriticalSection(flpCriticalSection);
      return kTRUE;
   }
   LeaveCriticalSection(flpCriticalSection);
   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGWin32::AllocColor(Colormap_t cmap, ColorStruct_t & color)
{
   // Find and allocate a color cell according to the color values specified
   // in the ColorStruct_t. If no cell could be allocated it returns kFALSE,
   // otherwise kTRUE.
   EnterCriticalSection(flpCriticalSection);

   int status;

   fThreadP.color.red = color.fRed;
   fThreadP.color.green = color.fGreen;
   fThreadP.color.blue = color.fBlue;

   fThreadP.pParam = (void *) cmap;
   fThreadP.iParam = FALSE;
   fThreadP.iParam1 = TRUE;
   PostThreadMessage(fIDThread, WIN32_GDK_COLORMAP_ALLOC_COLOR, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   color.fPixel = fThreadP.color.pixel;

   LeaveCriticalSection(flpCriticalSection);
   return kTRUE;                // status != 0 ? kTRUE : kFALSE;
}

//______________________________________________________________________________
void TGWin32::QueryColor(Colormap_t cmap, ColorStruct_t & color)
{
   // Fill in the primary color components for a specific pixel value.
   // On input fPixel should be set on return the fRed, fGreen and
   // fBlue components will be set.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.color.pixel = color.fPixel;

   fThreadP.pRet = NULL;
   fThreadP.pParam = (void *) fColormap;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_NEW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GdkColorContext *cc = (GdkColorContext *)fThreadP.pRet;

   fThreadP.pParam = (void *) cc;
   fThreadP.iParam = 1;
   PostThreadMessage(fIDThread, WIN32_GDK_COLOR_CONTEXT_QUERY_COLORS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   color.fPixel = fThreadP.color.pixel = RGB(fThreadP.color.red, fThreadP.color.green, fThreadP.color.blue);
   color.fRed = fThreadP.color.red;
   color.fGreen = fThreadP.color.green;
   color.fBlue = fThreadP.color.blue;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::FreeColor(Colormap_t cmap, ULong_t pixel)
{
   // Free color cell with specified pixel value.

   // FIXME: to be implemented.
}

//______________________________________________________________________________
Int_t TGWin32::EventsPending()
{
    // Returns number of pending events.
   EnterCriticalSection(flpCriticalSection);

    PostThreadMessage(fIDThread, WIN32_GDK_EVENTS_PENDING, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
    return fThreadP.iRet;
}

//______________________________________________________________________________
void TGWin32::NextEvent(Event_t & event)
{
   // Copies first pending event from event queue to Event_t structure
   // and removes event from queue. Not all of the event fields are valid
   // for each event type, except fType and fWindow.
   EnterCriticalSection(flpCriticalSection);

   GdkEvent *xev = NULL;

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_GET_EVENT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xev = (GdkEvent *) fThreadP.pRet;

   // fill in Event_t
   event.fType = kOtherEvent;   // bb add
   if (xev == NULL){
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

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

      ev.fSendEvent = kFALSE; //xev.any.send_event ? kTRUE : kFALSE;

      fThreadP.pParam = (void *) &xev;
      PostThreadMessage(fIDThread, WIN32_GDK_EVENT_GET_TIME, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      ev.fTime = fThreadP.iRet;

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
            ev.fUser[1] = 0; //xev.client.data.l[1];
            ev.fUser[2] = 0; //xev.client.data.l[2];
            ev.fUser[3] = 0; //xev.client.data.l[3];
            ev.fUser[4] = 0; //xev.client.data.l[4];
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
         fThreadP.pRet = NULL;
         fThreadP.Drawable = (GdkDrawable *) xev.key.window;
         PostThreadMessage(fIDThread, WIN32_GW_CHILD, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         HWND tmpwin = (HWND) fThreadP.pRet;
//         HWND tmpwin = GetWindow((HWND) GDK_DRAWABLE_XID(xev.key.window), GW_CHILD);
         if (tmpwin) {
             fThreadP.Drawable = (GdkDrawable *) tmpwin;
             PostThreadMessage(fIDThread, WIN32_GDK_XID_TABLE_LOOKUP, 0, 0L);  
             WaitForSingleObject(fThreadP.hThrSem, INFINITE);
             ev.fUser[0] = (ULong_t) fThreadP.lRet;
         }
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
         
         fThreadP.Drawable = (GdkDrawable *) xev.button.window;
         fThreadP.x = xev.button.x;
         fThreadP.y = xev.button.y;
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_CHILD_FROM_POINT, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         HWND tmpwin = (HWND)fThreadP.pRet;
         if (tmpwin) {
             fThreadP.Drawable = (GdkDrawable *) tmpwin;
             PostThreadMessage(fIDThread, WIN32_GDK_XID_TABLE_LOOKUP, 0, 0L);  
             WaitForSingleObject(fThreadP.hThrSem, INFINITE);
             ev.fUser[0] = (ULong_t) fThreadP.lRet;
         }
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

         fThreadP.pRet = NULL;
         fThreadP.Drawable = (GdkDrawable *) xev.motion.window;
         fThreadP.x = xev.button.x;
         fThreadP.y = xev.button.y;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_CHILD_FROM_POINT, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         HWND tmpwin = (HWND)fThreadP.pRet;
         if (tmpwin) {
             fThreadP.Drawable = (GdkDrawable *) tmpwin;
             PostThreadMessage(fIDThread, WIN32_GDK_XID_TABLE_LOOKUP, 0, 0L);  
             WaitForSingleObject(fThreadP.hThrSem, INFINITE);
             ev.fUser[0] = (ULong_t) fThreadP.lRet;
         }
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
      if (xev.type == GDK_SCROLL) {
         ev.fType = kButtonRelease;
         if (xev.scroll.direction == GDK_SCROLL_UP)
            ev.fCode = kButton4;
         else if (xev.scroll.direction == GDK_SCROLL_DOWN)
            ev.fCode = kButton5;
         ev.fWindow = (Window_t) xev.scroll.window;
         ev.fX = xev.scroll.x;
         ev.fY = xev.scroll.y;
         ev.fXRoot = xev.scroll.x_root;
         ev.fYRoot = xev.scroll.y_root;

         fThreadP.Drawable = (GdkDrawable *) xev.scroll.window;
         fThreadP.x = xev.scroll.x;
         fThreadP.y = xev.scroll.y;
         fThreadP.pRet = NULL;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_CHILD_FROM_POINT, 0, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
         HWND tmpwin = (HWND)fThreadP.pRet;
         if (tmpwin) {
             fThreadP.Drawable = (GdkDrawable *) tmpwin;
             PostThreadMessage(fIDThread, WIN32_GDK_XID_TABLE_LOOKUP, 0, 0L);  
             WaitForSingleObject(fThreadP.hThrSem, INFINITE);
             ev.fUser[0] = (ULong_t) fThreadP.lRet;
         }
         else
            ev.fUser[0] = (ULong_t) 0;
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::Bell(Int_t percent)
{
   EnterCriticalSection(flpCriticalSection);

   PostThreadMessage(fIDThread, WIN32_GDK_BEEP, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::CopyArea(Drawable_t src, Drawable_t dest, GContext_t gc,
                       Int_t src_x, Int_t src_y, UInt_t width,
                       UInt_t height, Int_t dest_x, Int_t dest_y)
{
   // Copy a drawable (i.e. pixmap) to another drawable (pixmap, window).
   // The graphics context gc will be used and the source will be copied
   // from src_x,src_y,src_x+width,src_y+height to dest_x,dest_y.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) dest;
   fThreadP.pParam = (void *) src;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.x = src_x;
   fThreadP.y = src_y;
   fThreadP.w = width;
   fThreadP.h = height;
   fThreadP.xpos = dest_x;
   fThreadP.ypos = dest_y;

   PostThreadMessage(fIDThread, WIN32_GDK_WIN_COPY_AREA, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ChangeWindowAttributes(Window_t id,
                                     SetWindowAttributes_t * attr)
{
   // Change window attributes.
   EnterCriticalSection(flpCriticalSection);

   GdkWMDecoration deco;
   GdkEventMask xevent_mask;
   UInt_t xevmask;
   Mask_t evmask;
   HWND w, flag;

   if (attr && (attr->fMask & kWAEventMask)) {
      evmask = (Mask_t) attr->fEventMask;
      MapEventMask(evmask, xevmask);
      fThreadP.Drawable = (GdkDrawable *) id;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_EVENTS, xevmask, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (attr && (attr->fMask & kWABackPixel)) {
      fThreadP.Drawable = (GdkDrawable *) id;
      fThreadP.color.pixel = attr->fBackgroundPixel;
      fThreadP.color.red = GetRValue(attr->fBackgroundPixel);
      fThreadP.color.green = GetGValue(attr->fBackgroundPixel);
      fThreadP.color.blue = GetBValue(attr->fBackgroundPixel);
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACKGROUND, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
//   if (attr && (attr->fMask & kWAOverrideRedirect))
//      gdk_window_set_override_redirect ((GdkWindow *) id, attr->fOverrideRedirect);
   if (attr && (attr->fMask & kWABackPixmap)) {
      fThreadP.Drawable = (GdkDrawable *) id;
      fThreadP.pParam = (void *) attr->fBackgroundPixmap;
      fThreadP.iParam = 0;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_BACK_PIXMAP, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (attr && (attr->fMask & kWACursor)) {
      fThreadP.Drawable = (GdkDrawable *) id;
      fThreadP.pParam = (void *) attr->fCursor;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_CURSOR, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (attr && (attr->fMask & kWAColormap)) {
      fThreadP.Drawable = (GdkDrawable *) id;
      fThreadP.pParam = (void *) attr->fColormap;
      PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_COLORMAP, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   if (attr && (attr->fMask & kWABorderWidth)) {
      if (attr->fBorderWidth > 0) {
         fThreadP.Drawable = (GdkDrawable *) id;
         PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_DECOR, GDK_DECOR_BORDER, 0L);  
         WaitForSingleObject(fThreadP.hThrSem, INFINITE);
      }
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ChangeProperty(Window_t id, Atom_t property, Atom_t type,
                             UChar_t * data, Int_t len)
{
   // This function alters the property for the specified window and
   // causes the X server to generate a PropertyNotify event on that
   // window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.ulParam = property;
   fThreadP.ulParam1 = type;
   fThreadP.iParam = 8;
   fThreadP.iParam1 = GDK_PROP_MODE_REPLACE;
   fThreadP.iParam2 = len;
//   fThreadP.pParam = data;
   sprintf(fThreadP.sParam,"%s",data);
   PostThreadMessage(fIDThread, WIN32_GDK_PROPERTY_CHANGE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawLine(Drawable_t id, GContext_t gc, Int_t x1, Int_t y1,
                       Int_t x2, Int_t y2)
{
   // Draw a line.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.x1 = x1;
   fThreadP.y1 = y1;
   fThreadP.x2 = x2;
   fThreadP.y2 = y2;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_LINE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);

}

//______________________________________________________________________________
void TGWin32::ClearArea(Window_t id, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Clear a window area to the bakcground color.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_CLEAR_AREA, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   LeaveCriticalSection(flpCriticalSection);
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
//   MapEvent(tev, fThreadP.event, kTRUE);
   MapEvent(tev, xev, kTRUE);
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
//   fThreadP.iParam = fThreadP.event.type;
   fThreadP.pParam = (void *) & xev;
   fThreadP.iParam = xev.type;
   PostThreadMessage(fIDThread, WIN32_GDK_CHECK_TYPED_WIN_EVENT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   Bool_t r = fThreadP.iRet;
   LeaveCriticalSection(flpCriticalSection);
   if (r)
      MapEvent(ev, xev, kFALSE);
//      MapEvent(ev, fThreadP.event, kFALSE);
   return r ? kTRUE : kFALSE;

}

//______________________________________________________________________________
void TGWin32::SendEvent(Window_t id, Event_t * ev)
{
   // Send event ev to window id.

   if (!ev)
      return;

   GdkEvent xev;

//   MapEvent(*ev, fThreadP.event, kTRUE);
   MapEvent(*ev, xev, kTRUE);
   fThreadP.pParam = (void *) & xev;
   EnterCriticalSection(flpCriticalSection);
   PostThreadMessage(fIDThread, WIN32_GDK_EVENT_PUT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
//   gdk_flush();

   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::WMDeleteNotify(Window_t id)
{
   // Tell WM to send message when window is closed via WM.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_WM_DELETE_NOTIFY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetKeyAutoRepeat(Bool_t on)
{
   // Turn key auto repeat on or off.
   EnterCriticalSection(flpCriticalSection);

   PostThreadMessage(fIDThread, WIN32_GDK_SET_KEY_AUTOREPEAT, on, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::GrabKey(Window_t id, Int_t keycode, UInt_t modifier,
                      Bool_t grab)
{
   // Establish passive grab on a certain key. That is, when a certain key
   // keycode is hit while certain modifier's (Shift, Control, Meta, Alt)
   // are active then the keyboard will be grabed for window id.
   // When grab is false, ungrab the keyboard for this key and modifier.
   EnterCriticalSection(flpCriticalSection);

   UInt_t xmod;
   GdkEventMask masque;

   MapModifierState(modifier, xmod);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.uiParam = xmod;
   PostThreadMessage(fIDThread, WIN32_GDK_GRAB_KEY, grab, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   UInt_t xevmask;
   GdkEventMask masque;
   UInt_t xmod;

   MapModifierState(modifier, xmod);
   MapEventMask(evmask, xevmask);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.uiParam = xevmask;
   PostThreadMessage(fIDThread, WIN32_GDK_GRAB_BUTTON, grab, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::GrabPointer(Window_t id, UInt_t evmask, Window_t confine,
                          Cursor_t cursor, Bool_t grab,
                          Bool_t owner_events)
{
   // Establish an active pointer grab. While an active pointer grab is in
   // effect, further pointer events are only reported to the grabbing
   // client window.
   EnterCriticalSection(flpCriticalSection);

   UInt_t xevmask;
   MapEventMask(evmask, xevmask);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.iParam = owner_events;
   fThreadP.uiParam = xevmask;
   fThreadP.pParam1 = (void *) confine;
   fThreadP.pParam2 = (void *) cursor;
   PostThreadMessage(fIDThread, WIN32_GDK_GRAB_POINTER, grab, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWindowName(Window_t id, char *name)
{
   // Set window name.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   sprintf(fThreadP.sParam,"%s",name);
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_TITLE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetIconName(Window_t id, char *name)
{
   // Set window icon name.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   sprintf(fThreadP.sParam,"%s",name);
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_ICON_NAME, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetIconPixmap(Window_t id, Pixmap_t pic)
{
   // Set pixmap the WM can use when the window is iconized.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *) pic;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_ICON, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

#define safestrlen(s) ((s) ? strlen(s) : 0)

//______________________________________________________________________________
void TGWin32::SetClassHints(Window_t id, char *className,
                            char *resourceName)
{
   // Set the windows class and resource name.
   EnterCriticalSection(flpCriticalSection);

   char *class_string;
   char *s;
   int len_nm, len_cl;
   GdkAtom type, prop;

   fThreadP.iParam = kFALSE;
   sprintf(fThreadP.sParam,"WM_CLASS");
   PostThreadMessage(fIDThread, WIN32_GDK_ATOM, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   prop = (GdkAtom)fThreadP.ulRet;

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
      fThreadP.Drawable = (GdkDrawable *) id;
      fThreadP.ulParam = XA_WM_CLASS;
      fThreadP.ulParam1 = XA_STRING;
      fThreadP.iParam = 8;
      fThreadP.iParam1 = GDK_PROP_MODE_REPLACE;
      fThreadP.iParam2 = len_nm + len_cl + 2;
      sprintf(fThreadP.sParam,"%s",class_string);
//      fThreadP.pParam = (void *) class_string;
      PostThreadMessage(fIDThread, WIN32_WIN32_PROPERTY_CHANGE, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
     
      free(class_string);
   }
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetMWMHints(Window_t id, UInt_t value, UInt_t funcs,
                          UInt_t input)
{
   // Set decoration style for MWM-compatible wm (mwm, ncdwm, fvwm?).
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_DECOR, value, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.iParam = funcs;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_FUNCTIONS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWMPosition(Window_t id, Int_t x, Int_t y)
{
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.x = x;
   fThreadP.y = y;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_MOVE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWMSize(Window_t id, UInt_t w, UInt_t h)
{
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.w = w;
   fThreadP.h = h;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_RESIZE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetWMSizeHints(Window_t id, UInt_t wmin, UInt_t hmin,
                             UInt_t wmax, UInt_t hmax,
                             UInt_t winc, UInt_t hinc)
{
   // Give the window manager minimum and maximum size hints. Also
   // specify via winc and hinc the resize increments.
   EnterCriticalSection(flpCriticalSection);

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

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.iParam = flags;
   fThreadP.pParam = (void *) &hints;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_GEOM_HINTS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.pParam = (void *) main_id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_TRANSIENT_FOR, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawString(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                         const char *s, Int_t len)
{
   // Draw a string using a specific graphics context in position (x,y).
   EnterCriticalSection(flpCriticalSection);

   GdkGCValues values;
   fThreadP.GC = (GdkGC *) gc;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.pParam = (void *) fThreadP.gcvals.font;
   fThreadP.x = x;
   fThreadP.y = y;
   sprintf(fThreadP.sParam,"%s",s);
   fThreadP.iParam = len;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_TEXT, 0, 0L);
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::TextWidth(FontStruct_t font, const char *s, Int_t len)
{
   // Return lenght of string in pixels. Size depends on font.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) font;
   fThreadP.iParam = len;
   sprintf(fThreadP.sParam,"%s",s);
   PostThreadMessage(fIDThread, WIN32_GDK_GET_TEXT_WIDTH, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return fThreadP.iRet;
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
   EnterCriticalSection(flpCriticalSection);

   GdkGCValues xgval;
   ULong_t xmask;

   MapGCValues(gval, xmask, xgval, kTRUE);

   fThreadP.GC = (GdkGC *) gc;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_GET_VALUES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   MapGCValues(gval, xmask, fThreadP.gcvals, kFALSE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
FontStruct_t TGWin32::GetFontStruct(FontH_t fh)
{
   // Retrieve associated font structure once we have the font handle.
   // Free returned FontStruct_t using FreeFontStruct().
   EnterCriticalSection(flpCriticalSection);

   GdkFont *fs;

   fThreadP.pRet = NULL;
   fThreadP.pParam = (void *) fh;
   PostThreadMessage(fIDThread, WIN32_GDK_FONT_REF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   fs = (GdkFont *)fThreadP.pRet;

   LeaveCriticalSection(flpCriticalSection);
   return (FontStruct_t) fs;
}

//______________________________________________________________________________
void TGWin32::FreeFontStruct(FontStruct_t fs)
{
   // Free font structure returned by GetFontStruct().
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) fs;
   PostThreadMessage(fIDThread, WIN32_GDK_FONT_UNREF, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::ClearWindow(Window_t id)
{
   // Clear window.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_CLEAR, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Int_t TGWin32::KeysymToKeycode(UInt_t keysym)
{
   // Convert a keysym to the appropriate keycode. For example keysym is
   // a letter and keycode is the matching keyboard key (which is dependend
   // on the current keyboard mapping).
   EnterCriticalSection(flpCriticalSection);

   UInt_t xkeysym;

   MapKeySym(keysym, xkeysym);
   fThreadP.uiParam = xkeysym;
   PostThreadMessage(fIDThread, WIN32_GDK_KEYVAL_FROM_NAME, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xkeysym = fThreadP.uiRet;
   LeaveCriticalSection(flpCriticalSection);
   return xkeysym;
   return 0;
}

//______________________________________________________________________________
void TGWin32::FillRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                            UInt_t w, UInt_t h)
{
   // Draw a filled rectangle. Filling is done according to the gc.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.bFill = kTRUE;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawRectangle(Drawable_t id, GContext_t gc, Int_t x, Int_t y,
                            UInt_t w, UInt_t h)
{
   // Draw a rectangle outline.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.w = w;
   fThreadP.h = h;
   fThreadP.bFill = kFALSE;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_RECTANGLE, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DrawSegments(Drawable_t id, GContext_t gc, Segment_t * seg,
                           Int_t nseg)
{
   // Draws multiple line segments. Each line is specified by a pair of points.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.pParam = (void *) seg;
   fThreadP.iParam = nseg;
   PostThreadMessage(fIDThread, WIN32_GDK_DRAW_SEGMENTS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SelectInput(Window_t id, UInt_t evmask)
{
   // Defines which input events the window is interested in. By default
   // events are propageted up the window stack. This mask can also be
   // set at window creation time via the SetWindowAttributes_t::fEventMask
   // attribute.
   EnterCriticalSection(flpCriticalSection);

   UInt_t xevmask;
   GdkEventMask masque;

   MapEventMask(evmask, xevmask, kTRUE);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.uiParam = xevmask;
   PostThreadMessage(fIDThread, WIN32_GDK_SELECT_INPUT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Window_t TGWin32::GetInputFocus()
{
   // Returns the window id of the window having the input focus.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_GET_INPUT_FOCUS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Window_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::SetInputFocus(Window_t id)
{
   // Set keyboard input focus to window id.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_SET_INPUT_FOCUS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Window_t TGWin32::GetPrimarySelectionOwner()
{
   // Returns the window id of the current owner of the primary selection.
   // That is the window in which, for example some text is selected.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_SELECTION_OWNER_GET, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Window_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::SetPrimarySelectionOwner(Window_t id)
{
   // Makes the window id the current owner of the primary selection.
   // That is the window in which, for example some text is selected.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.lParam = clipboard_atom;
   PostThreadMessage(fIDThread, WIN32_GDK_SELECTION_OWNER_SET, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.ulParam = clipboard;
   fThreadP.uiParam = when;
   PostThreadMessage(fIDThread, WIN32_GDK_SELECTION_CONVERT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   char *data;
   int nread, actual_format;

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.ulParam = atom;
   PostThreadMessage(fIDThread, WIN32_GDK_SELECTION_PROP_GET, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   nread = fThreadP.iRet;
   actual_format = fThreadP.iRet1;
   data = fThreadP.sRet;

//    printf("TGWin32::GetPasteBuffer : %s\n",data);
   if ((nread == 0) || (data == NULL)) {
      nchar = 0;
      LeaveCriticalSection(flpCriticalSection);
      return;
   }
   text.Insert(0, (const char *) data);
   nchar = 1;                   //strlen(data);
   g_free(data);
   if (del) {
      fThreadP.Drawable = (GdkDrawable *) id;
      PostThreadMessage(fIDThread, WIN32_GDK_PROP_DELETE, 0, 0L);  
      WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   }
   LeaveCriticalSection(flpCriticalSection);

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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) src;
   fThreadP.pParam = (void *) dest;
   fThreadP.x = src_x;
   fThreadP.y = src_y;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_TRANSLATE_COORDINATES, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   child = (Window_t) fThreadP.pRet;
   if (child == src)
      child = (Window_t) 0;
   dest_x = fThreadP.x1;
   dest_y = fThreadP.y1;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::GetWindowSize(Drawable_t id, Int_t & x, Int_t & y,
                            UInt_t & w, UInt_t & h)
{
   // Return geometry of window (should be called GetGeometry but signature
   // already used).
   EnterCriticalSection(flpCriticalSection);

   Int_t ddum;

   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   x = fThreadP.x;
   y = fThreadP.y;
   w = fThreadP.w;
   h = fThreadP.h;

   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.Drawable = (GdkDrawable *) id;
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.pParam = (GdkPoint *) points;
   fThreadP.iParam = npnt;

   PostThreadMessage(fIDThread, WIN32_GDK_FILL_POLYGON, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   LeaveCriticalSection(flpCriticalSection);
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
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   fThreadP.pRet1 = NULL;
   fThreadP.Drawable = (GdkDrawable *) id;
   PostThreadMessage(fIDThread, WIN32_GDK_QUERY_POINTER1, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   root_x = fThreadP.x;
   root_y = fThreadP.y;
   win_x = fThreadP.x1;
   win_y = fThreadP.y1;
   rootw = (Window_t) fThreadP.pRet;
   childw = (Window_t) fThreadP.pRet1;
   mask = 0L;
   MapEventMask(mask, fThreadP.uiRet, kFALSE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetForeground(GContext_t gc, ULong_t foreground)
{
   // Set foreground color in graphics context (shortcut for ChangeGC with
   // only foreground mask set).
   EnterCriticalSection(flpCriticalSection);

   fThreadP.GC = (GdkGC *) gc;
   fThreadP.color.pixel = foreground;
   fThreadP.color.red = GetRValue(foreground);
   fThreadP.color.green = GetGValue(foreground);
   fThreadP.color.blue = GetBValue(foreground);
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_FOREGROUND, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SetClipRectangles(GContext_t gc, Int_t x, Int_t y,
                                Rectangle_t * recs, Int_t n)
{
   // Set clipping rectangles in graphics context. X, Y specify the origin
   // of the rectangles. Recs specifies an array of rectangles that define
   // the clipping mask and n is the number of rectangles.
   EnterCriticalSection(flpCriticalSection);

   Int_t i;
   GdkRectangle *grects = new GdkRectangle[n];
   for (i = 0; i < n; i++) {
      grects[i].x = recs[i].fX;
      grects[i].y = recs[i].fY;
      grects[i].width = recs[i].fWidth;
      grects[i].height = recs[i].fHeight;
   }
   fThreadP.GC = (GdkGC *) gc;
   fThreadP.pParam = (void *) grects;
   PostThreadMessage(fIDThread, WIN32_GDK_GC_SET_CLIP_RECT, 1, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::Update(Int_t mode)
{
   // Flush (mode = 0, default) or synchronize (mode = 1) X output buffer.
   // Flush flushes output buffer. Sync flushes buffer and waits till all
   // requests have been processed by X server.
   EnterCriticalSection(flpCriticalSection);

   PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Region_t TGWin32::CreateRegion()
{
   // Create a new empty region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_NEW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Region_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::DestroyRegion(Region_t reg)
{
   // Destroy region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) reg;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_DESTROY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::UnionRectWithRegion(Rectangle_t * rect, Region_t src,
                                  Region_t dest)
{
   // Union of rectangle with a region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) src;
   fThreadP.x = rect->fX;
   fThreadP.y = rect->fY;
   fThreadP.w = rect->fWidth;
   fThreadP.h = rect->fHeight;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_UNION_WITH_RECT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   dest = (Region_t) fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Region_t TGWin32::PolygonRegion(Point_t * points, Int_t np, Bool_t winding)
{
   // Create region for the polygon defined by the points array.
   // If winding is true use WindingRule else EvenOddRule as fill rule.
   EnterCriticalSection(flpCriticalSection);

   int i;
   GdkPoint *xy = new GdkPoint[np];

   for (i = 0; i < np; i++) {
      xy[i].x = points[i].fX;
      xy[i].y = points[i].fY;
   }
   fThreadP.pParam = (void *) xy;
   fThreadP.iParam = np;
   fThreadP.iParam1 = winding;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_POLYGON, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Region_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::UnionRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Compute the union of rega and regb and return result region.
   // The output region may be the same result region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) rega;
   fThreadP.pParam1 = (void *)regb;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGIONS_UNION, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   result = (Region_t) fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::IntersectRegion(Region_t rega, Region_t regb,
                              Region_t result)
{
   // Compute the intersection of rega and regb and return result region.
   // The output region may be the same as the result region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) rega;
   fThreadP.pParam1 = (void *) regb;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGIONS_INTERSECT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   result = (Region_t) fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::SubtractRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Subtract rega from regb.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) rega;
   fThreadP.pParam1 = (void *) regb;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGIONS_SUBSTRACT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   result = (Region_t) fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::XorRegion(Region_t rega, Region_t regb, Region_t result)
{
   // Calculate the difference between the union and intersection of
   // two regions.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) rega;
   fThreadP.pParam1 = (void *) regb;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGIONS_XOR, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   result = (Region_t) fThreadP.pRet;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Bool_t TGWin32::EmptyRegion(Region_t reg)
{
   // Return true if the region is empty.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   fThreadP.pParam = (void *) reg;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_EMPTY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Bool_t) fThreadP.iRet;
}

//______________________________________________________________________________
Bool_t TGWin32::PointInRegion(Int_t x, Int_t y, Region_t reg)
{
   // Returns true if the point x,y is in the region.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) reg;
   fThreadP.x = x;
   fThreadP.y = y;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_POINT_IN, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Bool_t) fThreadP.iRet;
}

//______________________________________________________________________________
Bool_t TGWin32::EqualRegion(Region_t rega, Region_t regb)
{
   // Returns true if two regions are equal.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pParam = (void *) rega;
   fThreadP.pParam1 = (void *) regb;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_EQUAL, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   LeaveCriticalSection(flpCriticalSection);
   return (Bool_t) fThreadP.iRet;
}

//______________________________________________________________________________
void TGWin32::GetRegionBox(Region_t reg, Rectangle_t * rect)
{
   // Return smallest enclosing rectangle.
   EnterCriticalSection(flpCriticalSection);

   fThreadP.pRet = NULL;
   fThreadP.pParam = (void *) reg;
   PostThreadMessage(fIDThread, WIN32_GDK_REGION_GET_CLIPBOX, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   rect->fX = fThreadP.x;
   rect->fY = fThreadP.y;
   rect->fWidth = fThreadP.w;
   rect->fHeight = fThreadP.h;
   LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
char **TGWin32::ListFonts(char *fontname, Int_t /*max*/, Int_t &count)
{
    char **fontlist;
    Int_t fontcount = 0;
    EnterCriticalSection(flpCriticalSection);

    fThreadP.pRet = NULL;
    sprintf(fThreadP.sParam,"%s",fontname);
    PostThreadMessage(fIDThread, WIN32_GDK_FONTLIST_NEW, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    fontlist = (char **)fThreadP.pRet;
    fontcount = fThreadP.iRet;
    count = fontcount;
    LeaveCriticalSection(flpCriticalSection);
    if (fontcount > 0)
        return fontlist;
    return 0;
}

//______________________________________________________________________________
void TGWin32::FreeFontNames(char **fontlist)
{
    EnterCriticalSection(flpCriticalSection);
    fThreadP.pParam = fontlist;
    PostThreadMessage(fIDThread, WIN32_GDK_FONTLIST_FREE, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Drawable_t TGWin32::CreateImage(UInt_t width, UInt_t height)
{
    EnterCriticalSection(flpCriticalSection);

    fThreadP.w = width;
    fThreadP.h = height;
    fThreadP.pRet = NULL;
    PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_NEW, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    LeaveCriticalSection(flpCriticalSection);
    return (Drawable_t) fThreadP.pRet;
}

//______________________________________________________________________________
void TGWin32::GetImageSize(Drawable_t id, UInt_t &width, UInt_t &height)
{
    width  = ((GdkImage*)id)->width;
    height = ((GdkImage*)id)->height;
}

//______________________________________________________________________________
void TGWin32::PutPixel(Drawable_t id, Int_t x, Int_t y, ULong_t pixel)
{
   GdkImage *image = (GdkImage *)id;
   if (image->depth == 1)
      if (pixel & 1)
         ((UChar_t *) image->mem)[y * image->bpl + (x >> 3)] |= (1 << (7 - (x & 0x7)));
      else
         ((UChar_t *) image->mem)[y * image->bpl + (x >> 3)] &= ~(1 << (7 - (x & 0x7)));
   else {
      UChar_t *pixelp = (UChar_t *) image->mem + y * image->bpl + x * image->bpp;
      // Windows is always LSB, no need to check image->byte_order.
      switch (image->bpp) {
         case 4:
            pixelp[3] = 0;
         case 3:
            pixelp[2] = ((pixel >> 16) & 0xFF);
         case 2:
            pixelp[1] = ((pixel >> 8) & 0xFF);
         case 1:
            pixelp[0] = (pixel & 0xFF);
      }
   }
}

//______________________________________________________________________________
ULong_t TGWin32::GetPixel(Drawable_t id, Int_t x, Int_t y)
{
   GdkImage *image = (GdkImage *)id;
   ULong_t pixel;

   if (image->depth == 1)
      pixel = (((char *) image->mem)[y * image->bpl + (x >> 3)] & (1 << (7 - (x & 0x7)))) != 0;
   else {
      UChar_t *pixelp = (UChar_t *) image->mem + y * image->bpl + x * image->bpp;
      switch (image->bpp) {
         case 1:
            pixel = *pixelp;
            break;
         // Windows is always LSB, no need to check image->byte_order.
         case 2:
            pixel = pixelp[0] | (pixelp[1] << 8);
            break;
         case 3:
            pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
            break;
         case 4:
            pixel = pixelp[0] | (pixelp[1] << 8) | (pixelp[2] << 16);
            break;
      }
   }
   return pixel;
}
//______________________________________________________________________________
void TGWin32::PutImage(Drawable_t id, GContext_t gc, Drawable_t img, Int_t dx,
                       Int_t dy, Int_t x, Int_t y, UInt_t w, UInt_t h)
{
    EnterCriticalSection(flpCriticalSection);
    fThreadP.Drawable = (GdkDrawable *) id;
    fThreadP.GC = (GdkGC *)gc; 
    fThreadP.pParam = (GdkImage*)img;
    fThreadP.x = x;
    fThreadP.y = y;
    fThreadP.x1 = dx;
    fThreadP.y1 = dy;
    fThreadP.w = w;
    fThreadP.h = h;
    PostThreadMessage(fIDThread, WIN32_GDK_DRAW_IMAGE, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    PostThreadMessage(fIDThread, WIN32_GDK_FLUSH, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
void TGWin32::DeleteImage(Drawable_t img)
{
    EnterCriticalSection(flpCriticalSection);
    fThreadP.pParam = (void *)img;
    PostThreadMessage(fIDThread, WIN32_GDK_IMAGE_UNREF, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    LeaveCriticalSection(flpCriticalSection);
}

//______________________________________________________________________________
Window_t TGWin32::CreateGLWindow(Window_t wind, Visual_t visual, Int_t depth)
{
   // X11 specific code to initialize GL window.
   EnterCriticalSection(flpCriticalSection);

   GdkWindow *GLWin;
   int xval, yval;
   int wval, hval;

   fThreadP.Drawable = (GdkDrawable *) wind;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_GEOMETRY, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   xval  = fThreadP.x;
   yval  = fThreadP.y;
   wval  = fThreadP.w;
   hval  = fThreadP.h;

   // window attributes
   ULong_t mask;

   fThreadP.xattr.width = wval;
   fThreadP.xattr.height = hval;
   fThreadP.xattr.x = xval;
   fThreadP.xattr.y = yval;
   fThreadP.xattr.wclass = GDK_INPUT_OUTPUT;
   fThreadP.xattr.event_mask = 0L; //GDK_ALL_EVENTS_MASK;
   fThreadP.xattr.event_mask |= GDK_EXPOSURE_MASK | GDK_STRUCTURE_MASK | GDK_KEY_PRESS_MASK | GDK_KEY_RELEASE_MASK;
   fThreadP.xattr.colormap = gdk_colormap_get_system();
//   fThreadP.xattr.event_mask = 0;
   mask = GDK_WA_X | GDK_WA_Y | GDK_WA_COLORMAP | GDK_WA_WMCLASS | 
       GDK_WA_NOREDIR;

   fThreadP.xattr.window_type = GDK_WINDOW_CHILD;

   fThreadP.Drawable = (GdkDrawable *) wind;
   fThreadP.lParam = mask;
   fThreadP.pRet = NULL;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_NEW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   GLWin = (GdkWindow *) fThreadP.pRet;

   fThreadP.Drawable = (GdkDrawable *) GLWin;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SET_EVENTS, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);
   
   fThreadP.Drawable = (GdkDrawable *) GLWin;
   PostThreadMessage(fIDThread, WIN32_GDK_WIN_SHOW, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   fThreadP.Drawable = (GdkDrawable *) GLWin;
   PostThreadMessage(fIDThread, WIN32_GDK_INIT_PIXEL_FORMAT, 0, 0L);  
   WaitForSingleObject(fThreadP.hThrSem, INFINITE);

   if(fThreadP.iRet == -1)
      Error("InitGLWindow", "Barf! ChoosePixelFormat Failed");
   if(fThreadP.iRet == -2)
      Error("InitGLWindow", "Barf! SetPixelFormat Failed");
   LeaveCriticalSection(flpCriticalSection);
   return (Window_t)GLWin;
}

//______________________________________________________________________________
ULong_t TGWin32::GetWinDC(Window_t wind)
{
    EnterCriticalSection(flpCriticalSection);
    fThreadP.Drawable = (GdkDrawable *) wind;
    PostThreadMessage(fIDThread, WIN32_GDK_GET_WIN_DC, 0, 0L);  
    WaitForSingleObject(fThreadP.hThrSem, INFINITE);
    LeaveCriticalSection(flpCriticalSection);
    return (ULong_t) fThreadP.pRet;
}
