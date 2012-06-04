// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define NDEBUG

#include <algorithm>
#include <cassert>

#include <Cocoa/Cocoa.h>

#include "ROOTOpenGLView.h"
#include "QuartzWindow.h"
#include "CocoaUtils.h"
#include "KeySymbols.h"
#include "X11Events.h"
#include "TGClient.h"
#include "TGWindow.h"
#include "TList.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

namespace {

//Convert unichar (from [NSEvent characters]) into
//ROOT's key symbol (from KeySymbols.h).
template<typename T1, typename T2>
struct KeySymPair {
   T1 fFirst;
   T2 fSecond;
   
   bool operator < (const KeySymPair &rhs)const
   {
      return fFirst < rhs.fFirst;
   }
};

}

//______________________________________________________________________________
void MapUnicharToKeySym(unichar key, char *buf, Int_t /*len*/, UInt_t &rootKeySym)
{
   assert(buf != 0 && "MapUnicharToKeySym, buf parameter is null");

   static const KeySymPair<unichar, EKeySym> keyMap[] = {
        {NSEnterCharacter, kKey_Enter},
        {NSTabCharacter, kKey_Tab},
        {NSCarriageReturnCharacter, kKey_Return},
        {NSBackTabCharacter, kKey_Backtab},
        //WHYWHYWHY apple does not have a constant for escape????
        {27, kKey_Escape},
        {NSDeleteCharacter, kKey_Backspace},
        {NSUpArrowFunctionKey, kKey_Up},
        {NSDownArrowFunctionKey, kKey_Down},
        {NSLeftArrowFunctionKey, kKey_Left},
        {NSRightArrowFunctionKey, kKey_Right},
        {NSF1FunctionKey, kKey_F1},
        {NSF2FunctionKey, kKey_F2},
        {NSF3FunctionKey, kKey_F3},
        {NSF4FunctionKey, kKey_F4},
        {NSF5FunctionKey, kKey_F5},
        {NSF6FunctionKey, kKey_F6},
        {NSF7FunctionKey, kKey_F7},
        {NSF8FunctionKey, kKey_F8},
        {NSF9FunctionKey, kKey_F8},
        {NSF10FunctionKey, kKey_F10},
        {NSF11FunctionKey, kKey_F11},
        {NSF12FunctionKey, kKey_F12},
        {NSF13FunctionKey, kKey_F13},
        {NSF14FunctionKey, kKey_F14},
        {NSF15FunctionKey, kKey_F15},
        {NSF16FunctionKey, kKey_F16},
        {NSF17FunctionKey, kKey_F17},
        {NSF18FunctionKey, kKey_F18},
        {NSF19FunctionKey, kKey_F19},
        {NSF20FunctionKey, kKey_F20},
        {NSF21FunctionKey, kKey_F21},
        {NSF22FunctionKey, kKey_F22},
        {NSF23FunctionKey, kKey_F23},
        {NSF24FunctionKey, kKey_F24},
        {NSF25FunctionKey, kKey_F25},
        {NSF26FunctionKey, kKey_F26},
        {NSF27FunctionKey, kKey_F27},
        {NSF28FunctionKey, kKey_F28},
        {NSF29FunctionKey, kKey_F29},
        {NSF30FunctionKey, kKey_F30},
        {NSF31FunctionKey, kKey_F31},
        {NSF32FunctionKey, kKey_F32},
        {NSF33FunctionKey, kKey_F33},
        {NSF34FunctionKey, kKey_F34},
        {NSF35FunctionKey, kKey_F35},
        {NSInsertFunctionKey, kKey_Insert},
        {NSDeleteFunctionKey, kKey_Delete},
        {NSHomeFunctionKey, kKey_Home},
        {NSEndFunctionKey, kKey_End},
        {NSPageUpFunctionKey, kKey_PageUp},
        {NSPageDownFunctionKey, kKey_PageDown},
        {NSPrintScreenFunctionKey, kKey_Print},
        {NSScrollLockFunctionKey, kKey_ScrollLock},
        {NSPauseFunctionKey, kKey_Pause},
        {NSSysReqFunctionKey, kKey_SysReq}};
   
   const unsigned nEntries = sizeof keyMap / sizeof keyMap[0];
   
   buf[1] = 0;

   KeySymPair<unichar, EKeySym> valueToFind = {};
   valueToFind.fFirst = key;
   const KeySymPair<unichar, EKeySym> *iter = std::lower_bound(keyMap, keyMap + nEntries, valueToFind);
   
   if (iter != keyMap + nEntries && iter->fFirst == key) {
      buf[0] = 0;
      rootKeySym = iter->fSecond;
   } else {
      buf[0] = key;//????
      rootKeySym = key;   
   }
}

//______________________________________________________________________________
Int_t MapKeySymToKeyCode(Int_t keySym)
{
   static const KeySymPair<EKeySym, unichar> keyMap[] = {
      {kKey_Escape, 27},
      {kKey_Tab, NSTabCharacter},
      {kKey_Backtab, NSBackTabCharacter},
      {kKey_Backspace, NSDeleteCharacter},
      {kKey_Return, NSCarriageReturnCharacter},
      {kKey_Enter, NSEnterCharacter},
      {kKey_Insert, NSInsertFunctionKey},
      {kKey_Delete, NSDeleteFunctionKey},
      {kKey_Pause, NSPauseFunctionKey},
      {kKey_Print, NSPrintScreenFunctionKey},
      {kKey_SysReq, NSSysReqFunctionKey},
      {kKey_Home, NSHomeFunctionKey},
      {kKey_End, NSEndFunctionKey},
      {kKey_Left, NSLeftArrowFunctionKey},
      {kKey_Up, NSUpArrowFunctionKey},
      {kKey_Right, NSRightArrowFunctionKey},
      {kKey_Down, NSDownArrowFunctionKey},
      {kKey_PageUp, NSPageUpFunctionKey},
      {kKey_PageDown, NSPageDownFunctionKey},
      //This part is bad.
      {kKey_Shift, 0},
      {kKey_Control, 0},
      {kKey_Alt, 0},
      {kKey_CapsLock, 0},
      {kKey_NumLock, 0},
      //
      {kKey_ScrollLock, NSScrollLockFunctionKey},
      {kKey_F1, NSF1FunctionKey},
      {kKey_F2, NSF2FunctionKey},
      {kKey_F3, NSF3FunctionKey},
      {kKey_F4, NSF4FunctionKey},
      {kKey_F5, NSF5FunctionKey},
      {kKey_F6, NSF6FunctionKey},
      {kKey_F7, NSF7FunctionKey},
      {kKey_F8, NSF8FunctionKey},
      {kKey_F8, NSF9FunctionKey},
      {kKey_F10, NSF10FunctionKey},
      {kKey_F11, NSF11FunctionKey},
      {kKey_F12, NSF12FunctionKey},
      {kKey_F13, NSF13FunctionKey},
      {kKey_F14, NSF14FunctionKey},
      {kKey_F15, NSF15FunctionKey},
      {kKey_F16, NSF16FunctionKey},
      {kKey_F17, NSF17FunctionKey},
      {kKey_F18, NSF18FunctionKey},
      {kKey_F19, NSF19FunctionKey},
      {kKey_F20, NSF20FunctionKey},
      {kKey_F21, NSF21FunctionKey},
      {kKey_F22, NSF22FunctionKey},
      {kKey_F23, NSF23FunctionKey},
      {kKey_F24, NSF24FunctionKey},
      {kKey_F25, NSF25FunctionKey},
      {kKey_F26, NSF26FunctionKey},
      {kKey_F27, NSF27FunctionKey},
      {kKey_F28, NSF28FunctionKey},
      {kKey_F29, NSF29FunctionKey},
      {kKey_F30, NSF30FunctionKey},
      {kKey_F31, NSF31FunctionKey},
      {kKey_F32, NSF32FunctionKey},
      {kKey_F33, NSF33FunctionKey},
      {kKey_F34, NSF34FunctionKey},
      {kKey_F35, NSF35FunctionKey}
   };

   const unsigned nEntries = sizeof keyMap / sizeof keyMap[0];

   KeySymPair<EKeySym, unichar> valueToFind = {};
   valueToFind.fFirst = static_cast<EKeySym>(keySym);
   const KeySymPair<EKeySym, unichar> *iter = std::lower_bound(keyMap, keyMap + nEntries, valueToFind);   
   if (iter != keyMap + nEntries && iter->fFirst == keySym)
      return iter->fSecond;

   return 0;
}

//______________________________________________________________________________
NSUInteger GetCocoaKeyModifiersFromROOTKeyModifiers(UInt_t rootModifiers)
{
   NSUInteger cocoaModifiers = 0;

   if (rootModifiers & kKeyLockMask)
      cocoaModifiers |= NSAlphaShiftKeyMask;
   if (rootModifiers & kKeyShiftMask)
      cocoaModifiers |= NSShiftKeyMask;
   if (rootModifiers & kKeyControlMask)
      cocoaModifiers |= NSControlKeyMask;
   if (rootModifiers & kKeyMod1Mask)
      cocoaModifiers |= NSAlternateKeyMask;
   if (rootModifiers & kKeyMod2Mask)
      cocoaModifiers |= NSCommandKeyMask;
   
   return cocoaModifiers;
}

namespace Detail {

//Several aux. functions to extract parameters from Cocoa events.

//______________________________________________________________________________
Time_t TimeForCocoaEvent(NSEvent *theEvent)
{
   //1. Event is not nil.
   assert(theEvent != nil && "TimeForCocoaEvent, event parameter is nil");

   return [theEvent timestamp] * 1000;//TODO: check this!
}

//______________________________________________________________________________
Event_t NewX11EventFromCocoaEvent(unsigned windowID, NSEvent *theEvent)
{
   //1. Event is not nil.

   assert(theEvent != nil && "NewX11EventFromCocoaEvent, event parameter is nil");

   Event_t newEvent = {};
   newEvent.fWindow = windowID;
   newEvent.fTime = TimeForCocoaEvent(theEvent);
   return newEvent;
}

//______________________________________________________________________________
void ConvertEventLocationToROOTXY(NSEvent *cocoaEvent, NSView<X11Window> *eventView, Event_t *rootEvent)
{
   //1. All parameters are valid.
   //Both event and view must be in the same window, I do not check this here.

   assert(cocoaEvent != nil && "ConvertEventLocationToROOTXY, cocoaEvent parameter is nil");
   assert(eventView != nil && "ConvertEventLocationToROOTXY, eventView parameter is nil");
   assert(rootEvent != 0 && "ConvertEventLocationToROOTXY, rootEvent parameter is null");

   //TODO: can [event window] be nil? (this can probably happen with mouse grabs).
   if (![cocoaEvent window])
      NSLog(@"Error in ConvertEventLocationToROOTXY, window property of event is nil, can not convert coordinates correctly");
   
   const NSPoint screenPoint = [[cocoaEvent window] convertBaseToScreen : [cocoaEvent locationInWindow]];
   NSPoint viewPoint = [[eventView window] convertScreenToBase : screenPoint];
   viewPoint = [eventView convertPointFromBase : viewPoint];

   rootEvent->fX = viewPoint.x;
   rootEvent->fY = viewPoint.y;

   WindowAttributes_t attr = {};
   GetRootWindowAttributes(&attr);
   
   rootEvent->fXRoot = screenPoint.x;
   rootEvent->fYRoot = attr.fHeight - screenPoint.y;
}

//______________________________________________________________________________
unsigned GetKeyboardModifiersFromCocoaEvent(NSEvent *theEvent)
{
   assert(theEvent != nil && "GetKeyboardModifiersFromCocoaEvent, event parameter is nil");

   const NSUInteger modifiers = [theEvent modifierFlags];
   unsigned rootModifiers = 0;
   if (modifiers & NSAlphaShiftKeyMask)
      rootModifiers |= kKeyLockMask;
   if (modifiers & NSShiftKeyMask)
      rootModifiers |= kKeyShiftMask;
   if (modifiers & NSControlKeyMask)
      rootModifiers |= kKeyControlMask;
   if (modifiers & NSAlternateKeyMask)
      rootModifiers |= kKeyMod1Mask;
   if (modifiers & NSCommandKeyMask)
      rootModifiers |= kKeyMod2Mask;

   return rootModifiers;
}

//______________________________________________________________________________
unsigned GetModifiersFromCocoaEvent(NSEvent *theEvent)
{
   assert(theEvent != nil && "GetModifiersFromCocoaEvent, event parameter is nil");

   unsigned rootModifiers = GetKeyboardModifiersFromCocoaEvent(theEvent);
   const NSUInteger buttons = [NSEvent pressedMouseButtons];
   if (buttons & 1)
      rootModifiers |= kButton1Mask;
   if (buttons & 2)
      rootModifiers |= kButton2Mask;

   return rootModifiers;
}

//Misc. aux. functions.

//______________________________________________________________________________
bool IsParent(NSView<X11Window>  *testParent, NSView<X11Window>  *testChild)
{
   assert(testParent != nil && "IsParent, testParent parameter is nil");
   assert(testChild != nil && "IsParent, testChild parameter is nil");

   if (testChild.fParentView) {
      NSView<X11Window> *parent = testChild.fParentView;
      while (parent) {
         if(parent == testParent)
            return true;
         parent = parent.fParentView;
      }
   }

   return false;
}

//______________________________________________________________________________
void BuildAncestryBranch(NSView<X11Window> *view, std::vector<NSView<X11Window> *> &branch)
{
   assert(view != nil && "BuildAncestryBranch, view parameter is nil");
   assert(view.fParentView != nil && "BuildAncestryBranch, view must have a parent");
   assert(view.fLevel > 0 && "BuildAncestryBranch, view has nested level 0");

   branch.resize(view.fLevel);
   
   NSView<X11Window>  *parent = view.fParentView;
   std::vector<NSView<X11Window> *>::reverse_iterator iter = branch.rbegin(), endIter = branch.rend();
   for (; iter != endIter; ++iter) {
      assert(parent != nil && "BuildAncestryBranch, fParentView is nil");
      *iter = parent;
      parent = parent.fParentView;
   }
}

//______________________________________________________________________________
Ancestry FindLowestCommonAncestor(NSView<X11Window> *view1, std::vector<NSView<X11Window> *> &branch1, 
                                  NSView<X11Window> *view2, std::vector<NSView<X11Window> *> &branch2, 
                                  NSView<X11Window> **lca)
{
   //Search for the lowest common ancestor.
   //View1 can not be parent of view2, view2 can not be parent of view1,
   //I do not check this condition here.

   assert(view1 != nil && "FindLowestCommonAncestor, view1 parameter is nil");
   assert(view2 != nil && "findLowestCommonAncestor, view2 parameter is nil");
   assert(lca != 0 && "FindLowestCommonAncestor, lca parameter is null");
   
   if (!view1.fParentView)
      return kAAncestorIsRoot;

   if (!view2.fParentView)
      return kAAncestorIsRoot;
   
   BuildAncestryBranch(view1, branch1);
   BuildAncestryBranch(view2, branch2);
   
   NSView<X11Window> *ancestor = nil;
   
   for (unsigned i = 0, j = 0; i < view1.fLevel && j < view2.fLevel && branch1[i] == branch2[j]; ++i, ++j)
      ancestor = branch1[i];

   if (ancestor) {
      *lca = ancestor;
      return kAHaveNonRootAncestor;
   }
   
   return kAAncestorIsRoot;
}

//______________________________________________________________________________
NSView<X11Window> *FindViewToPropagateEvent(NSView<X11Window> *viewFrom, Mask_t checkMask)
{
   //This function does not check passive grabs.
   assert(viewFrom != nil && "FindViewToPropagateEvent, view parameter is nil");
   
   if (viewFrom.fEventMask & checkMask)
      return viewFrom;
   
   for (viewFrom = viewFrom.fParentView; viewFrom; viewFrom = viewFrom.fParentView) {
      if (viewFrom.fEventMask & checkMask)
         return viewFrom;
   }

   return nil;
}

//______________________________________________________________________________
NSView<X11Window> *FindViewToPropagateEvent(NSView<X11Window> *viewFrom, Mask_t checkMask, NSView<X11Window> *grabView, Mask_t grabMask)
{
   //This function does not check passive grabs.
   assert(viewFrom != nil && "FindViewToPropagateEvent, view parameter is nil");
   
   if (viewFrom.fEventMask & checkMask)
      return viewFrom;
   
   for (viewFrom = viewFrom.fParentView; viewFrom; viewFrom = viewFrom.fParentView) {
      if (viewFrom.fEventMask & checkMask)
         return viewFrom;
   }
   
   if (grabView && (grabMask & checkMask))
      return grabView;

   return nil;
}

//Aux. 'low-level' functions to generate events and call HandleEvent for a root window.

//______________________________________________________________________________
bool IsMaskedEvent(EGEventType type)
{
   return type == kButtonPress || type == kButtonRelease || type == kGKeyPress || type == kKeyRelease ||
          type == kEnterNotify || type == kLeaveNotify || type == kMotionNotify;
}

//______________________________________________________________________________
void SendEventWithFilter(TGWindow *window, Event_t &event)
{
   assert(window != 0 && "SendEventWithFilter, window parameter is null");

   //This code is taken from TGClient class, it's a ROOT's way
   //to implement modal loop: gClient enters a nested loop and waits
   //for UnmapNotify or DestroyNotify on a special window to
   //exist this loop. During modal loop, gClient also "filters"
   //events - only events for registered pop-ups or "waitforwindow"
   //are handled.


   //Comment from TGClient:
   //Handle masked events only if window wid is the window for which the
   //event was reported or if wid is a parent of the event window. The not
   //masked event are handled directly. The masked events are:
   //kButtonPress, kButtonRelease, kKeyPress, kKeyRelease, kEnterNotify,
   //kLeaveNotify, kMotionNotify.

   //From TGClient.
   //Emit signal for event recorder(s)
   if (event.fType != kConfigureNotify) {
      //gClient->ProcessedEvent(&event, window->GetId());
   }

   //This loop is from TGClient. Why window without parent can not handle event
   //(and be "waitforwindow" - I do not know).
   const bool maskedEvent = IsMaskedEvent(event.fType);
   
   for (TGWindow *ptr = window; ptr->GetParent() != 0; ptr = (TGWindow *) ptr->GetParent()) {
      if (ptr->GetId() == gClient->GetWaitForWindow() || !maskedEvent) {
         window->HandleEvent(&event);
         //Actually, this can never happen now, but may change in future, so I have this check here.
         if (event.fType == gClient->GetWaitForEvent() && event.fWindow == gClient->GetWaitForWindow())
            gClient->SetWaitForWindow(kNone);

         return;
      }
   }

   //This is the second loop (with nested loop) from TGClient.
   //check if this is a popup menu
   if (TList *lst = gClient->GetListOfPopups()) {
      TIter next(lst);
   
      while (TGWindow *popup = (TGWindow *)next()) {
         for (TGWindow *ptr = window; ptr->GetParent() != 0; ptr = (TGWindow *) ptr->GetParent()) {
            if (ptr->GetId() == popup->GetId() && maskedEvent) {
               window->HandleEvent(&event);
               
               //Actually, this can never happen now, but may change in future, so I have this check here.
               if (event.fType == gClient->GetWaitForEvent() && event.fWindow == gClient->GetWaitForWindow())
                  gClient->SetWaitForWindow(kNone);
               
               return;
            }
         }
      }   
   }
}

//______________________________________________________________________________
void SendEvent(TGWindow *window, Event_t &event)
{
   //Event parameter is non-const, it can go to gClient->ProcessedEvent, which
   //accepts non-const.

   assert(window != 0 && "SendEvent, window parameter is null");

   if (gClient->GetWaitForWindow() == kNone)
      window->HandleEvent(&event);
   else
      SendEventWithFilter(window, event);
}

//______________________________________________________________________________
void SendEnterEvent(NSView<X11Window> *view, NSEvent *theEvent, EXMagic detail)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //This view must receive enter notify, I do not check it here.

   assert(view != nil && "SendEnterEvent, view parameter is nil");
   assert(theEvent != nil && "SendEnterEvent, event parameter is nil");
   assert(view.fID != 0 && "SendEnterEvent, view.fID is 0");

   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendEnterEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t enterEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   enterEvent.fType = kEnterNotify;
   enterEvent.fCode = detail;
   enterEvent.fState = GetModifiersFromCocoaEvent(theEvent);
   //Coordinates. Event possible happend not in a view,
   //but window should be the same. Also, coordinates are always
   //inside a view.
   
   ConvertEventLocationToROOTXY(theEvent, view, &enterEvent);
   

   //Dispatch:
//   window->HandleEvent(&enterEvent);
   SendEvent(window, enterEvent);
}

//______________________________________________________________________________
void SendLeaveEvent(NSView<X11Window> *view, NSEvent *theEvent, EXMagic detail)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //This window should receive leave event, I do not check it here.

   assert(view != nil && "SendLeaveEvent, view parameter is nil");
   assert(theEvent != nil && "SendLeaveEvent, event parameter is nil");
   assert(view.fID != 0 && "SendLeaveEvent, view.fID is 0");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendLeaveEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t leaveEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   leaveEvent.fType = kLeaveNotify;
   leaveEvent.fCode = detail;
   leaveEvent.fState = GetModifiersFromCocoaEvent(theEvent);
   //Coordinates. Event possibly happend not in a view, also, coordinates are out of
   //the view.
   ConvertEventLocationToROOTXY(theEvent, view, &leaveEvent);
   //Dispatch:
//   window->HandleEvent(&leaveEvent);
   SendEvent(window, leaveEvent);
}

//______________________________________________________________________________
void SendPointerMotionEvent(NSView<X11Window> *view, NSEvent *theEvent)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View receives pointer motion events, I do not check this condition here.
   
   assert(view != nil && "SendPointerMotionEvent, view parameter is nil");
   assert(theEvent != nil && "SendPointerMotionEvent, event parameter is nil");
   assert(view.fID != 0 && "SendPointerMotionEvent, view.fID is 0");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendPointerMotionEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }
   
   Event_t motionEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   motionEvent.fType = kMotionNotify;
   motionEvent.fState = GetModifiersFromCocoaEvent(theEvent);//GetKeyboardModifiersFromCocoaEvent(theEvent);
   
   //TODO: motionEvent.fUser[0] = find subwindow.
   
   ConvertEventLocationToROOTXY(theEvent, view, &motionEvent);
   //Dispatch:
   //window->HandleEvent(&motionEvent);
   SendEvent(window, motionEvent);
}

//______________________________________________________________________________
void SendButtonPressEvent(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View receives this event (either grab or select input) 
   //   - I do not check this condition here.

   assert(view != nil && "SendButtonPressEvent, view parameter is nil");
   assert(theEvent != nil && "SendButtonPressEvent, event parameter is nil");
   assert(view.fID != 0 && "SendButtonPressEvent, view.fID is 0");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendButtonpressEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t pressEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   pressEvent.fType = kButtonPress;
   pressEvent.fCode = btn;
   pressEvent.fState = GetKeyboardModifiersFromCocoaEvent(theEvent);
   //
   ConvertEventLocationToROOTXY(theEvent, view, &pressEvent);
   //
   //
   //ROOT uses "subwindow" parameter for button press event also, for example,
   //scroll bar has several children windows - "buttons", they are not selecting
   //button press events (and not grabbing pointer).
   //This will work wrong, if we have overlapping views - we'll find a wrong subwindow.
   //

   NSPoint viewPoint = {};
   viewPoint.x = pressEvent.fX;
   viewPoint.y = pressEvent.fY;
   for (NSView<X11Window> *child in [view subviews]) {
      if (!child.fIsOverlapped && [child hitTest : viewPoint]) {//Hit test goes down along the tree.
         pressEvent.fUser[0] = child.fID;
         break;
      }
   }
   
   //Dispatch:
   SendEvent(window, pressEvent);
}

//______________________________________________________________________________
void SendButtonReleaseEvent(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View must button release events, I do not check this here.

   assert(view != nil && "SendButtonReleaseEvent, view parameter is nil");
   assert(theEvent != nil && "SendButtonReleaseEvent, event parameter is nil");
   assert(view.fID != 0 && "SendButtonReleaseEvent, view.fID is 0");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendButtonReleaseEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }
 
   Event_t releaseEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   releaseEvent.fType = kButtonRelease;
   releaseEvent.fCode = btn;
   releaseEvent.fState = GetKeyboardModifiersFromCocoaEvent(theEvent);
   //
   ConvertEventLocationToROOTXY(theEvent, view, &releaseEvent);
   //Dispatch:
   SendEvent(window, releaseEvent);
}

//______________________________________________________________________________
void SendKeyPressEvent(NSView<X11Window> *view, NSView<X11Window> *childView, NSEvent *theEvent, NSPoint windowPoint)
{
   assert(view != nil && "SendKeyPressEvent, view parameter is nil");
   assert(theEvent != nil && "SendKeyPressEvent, event parameter is nil");
   assert(view.fID != 0 && "SendKeyPressEvent, view.fID is 0");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendKeyPressEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }
   
   Event_t keyPressEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   keyPressEvent.fType = kGKeyPress;
   keyPressEvent.fState = GetKeyboardModifiersFromCocoaEvent(theEvent);
   
   NSString *characters = [theEvent charactersIgnoringModifiers];
   assert(characters != nil && "SendKeyPressEvent, [theEvent characters] returned nil");
   assert([characters length] > 0 && "SendKeyPressEvent, characters is an empty string");

   keyPressEvent.fCode = [characters characterAtIndex : 0];
   
   const NSPoint viewPoint = [view convertPointFromBase : windowPoint];
   //Coords.
   keyPressEvent.fX = viewPoint.x;
   keyPressEvent.fY = viewPoint.y;
   const NSPoint screenPoint = TranslateToScreen(view, viewPoint);
   keyPressEvent.fXRoot = screenPoint.x;
   keyPressEvent.fYRoot = screenPoint.y;
   //Subwindow.
   if (childView)
      keyPressEvent.fUser[0] = childView.fID;
   
   SendEvent(window, keyPressEvent);
}

//______________________________________________________________________________
void SendFocusInEvent(NSView<X11Window> *view, EXMagic mode)
{
   assert(view != nil && "SendFocusInEvent, view parameter is nil");
   //
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendFocusInEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t focusInEvent = {};
   focusInEvent.fType = kFocusIn;
   focusInEvent.fCode = mode;
//   focusInEvent.fState = ;

   SendEvent(window, focusInEvent);
}

//______________________________________________________________________________
void SendFocusOutEvent(NSView<X11Window> *view, EXMagic mode)
{
   assert(view != nil && "SendFocusOutEvent, view parameter is nil");
   //
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendFocusOutEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t focusOutEvent = {};
   focusOutEvent.fType = kFocusOut;
   focusOutEvent.fCode = mode;//code mode :)
   //focusOutEvent.fState = ;
   
   SendEvent(window, focusOutEvent);
}

//Aux. functions to send events to view's branch.

//______________________________________________________________________________
void SendEnterEventRange(NSView<X11Window> *from, NSView<X11Window> *to, NSEvent *theEvent, EXMagic mode)
{
   //[from, to) - legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendEnterEventRange, 'from' parameter is nil");
   assert(to != nil && "SendEnterEventRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendEnterEventRange, event parameter is nil");
   
   while (from != to) {
      if (from.fEventMask & kEnterWindowMask)
         SendEnterEvent(from, theEvent, mode);
      from = from.fParentView;
   }
}

//______________________________________________________________________________
void SendEnterEventClosedRange(NSView<X11Window> *from, NSView<X11Window> *to, NSEvent *theEvent, EXMagic mode)
{
   //[from, to] - inclusive, legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendEnterEventClosedRange, 'from' parameter is nil");
   assert(to != nil && "SendEnterEventClosedRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendEnterEventClosedRange, event parameter is nil");
   
   SendEnterEventRange(from, to, theEvent, mode);
   if (to.fEventMask & kEnterWindowMask)
      SendEnterEvent(to, theEvent, mode);
}

//______________________________________________________________________________
void SendLeaveEventRange(NSView<X11Window> *from, NSView<X11Window> *to, NSEvent *theEvent, EXMagic mode)
{
   //[from, to) - legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendLeaveEventRange, 'from' parameter is nil");
   assert(to != nil && "SendLeaveEventRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendLeaveEventRange, event parameter is nil");

   while (from != to) {
      if (from.fEventMask & kLeaveWindowMask)
         SendLeaveEvent(from, theEvent, mode);
      from = from.fParentView;
   }
}

//______________________________________________________________________________
void SendLeaveEventClosedRange(NSView<X11Window> *from, NSView<X11Window> *to, NSEvent *theEvent, EXMagic mode)
{
   //[from, to] - inclusive, legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendLeaveEventClosedRange, 'from' parameter is nil");
   assert(to != nil && "SendLeaveEventClosedRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendLeaveEventClosedRange, event parameter is nil");

   SendLeaveEventRange(from, to, theEvent, mode);
   if (to.fEventMask & kLeaveWindowMask)
      SendLeaveEvent(to, theEvent, mode);
}

//Top-level crossing event generators.

//______________________________________________________________________________
void GenerateCrossingEventChildToParent(NSView<X11Window> *parent, NSView<X11Window> *child, NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and A is an inferior of B.
   //Generate LeaveNotify on A (with detail NotifyAncestor).
   //Generate LeaveNotify for every window between A and B, exclusive (with detail NotifyVirtual)
   //Generate EnterNotify for B with detail NotifyInferior.
   
   //ROOT does not have NotifyAncestor/NotifyInferior.
   
   assert(parent != nil && "GenerateCrossingEventChildToParent, parent parameter is nil");
   assert(child != nil && "GenerateCrossingEventChildToParent, child parameter is nil");
   assert(theEvent != nil && "GenerateCrossingEventChildToParent, event parameter is nil");
   assert(child.fParentView != nil && "GenerateCrossingEventChildToParent, child parameter must have QuartzView* parent");
   
   if (child.fEventMask & kLeaveWindowMask)
      SendLeaveEvent(child, theEvent, detail);

   SendLeaveEventRange(child.fParentView, parent, theEvent, detail);
   
   if (parent.fEventMask & kEnterWindowMask)
      SendEnterEvent(parent, theEvent, detail);
}

//______________________________________________________________________________
void GenerateCrossingEventParentToChild(NSView<X11Window> *parent, NSView<X11Window> *child, NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and B is an inferior of A.
   //Generate LeaveNotify event for A, detail == NotifyInferior.
   //Generate EnterNotify for each window between window A and window B, exclusive, detail == NotifyVirtual (no such entity in ROOT).
   //Generate EnterNotify on window B, detail == NotifyAncestor.
   
   //ROOT does not have NotifyInferior/NotifyAncestor.
   
   assert(parent != nil && "GenerateCrossingEventParentToChild, parent parameter is nil");
   assert(child != nil && "GenerateCrossingEventParentToChild, child parameter is nil");
   assert(theEvent != nil && "GenerateCrossingEventParentToChild, event parameter is nil");
   assert(child.fParentView != nil && "GenerateCrossingEventParentToChild, child parameter must have QuartzView* parent");
   
   if (parent.fEventMask & kLeaveWindowMask)
      SendLeaveEvent(parent, theEvent, detail);

   //I do not know, if the order must be reversed, but if yes - it's already FAR TOO
   //expensive to do (but I'll reuse my 'branch' arrays from  FindLowestAncestor).
   SendEnterEventRange(child.fParentView, parent, theEvent, detail);
   
   if (child.fEventMask & kEnterWindowMask)
      SendEnterEvent(child, theEvent, detail);
}

//______________________________________________________________________________
void GenerateCrossingEventFromChild1ToChild2(NSView<X11Window> *child1, NSView<X11Window> *child2, NSView<X11Window> *ancestor, NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and window C is their lowest common ancestor.
   //Generate LeaveNotify for window A with detail == NotifyNonlinear.
   //Generate LeaveNotify for each window between A and C, exclusive, with detail == NotifyNonlinearVirtual
   //Generate EnterNotify (detail == NotifyNonlinearVirtual) for each window between C and B, exclusive
   //Generate EnterNotify for window B, with detail == NotifyNonlinear.
   assert(child1 != nil && "GenerateCrossingEventFromChild1ToChild2, child1 parameter is nil");
   assert(child2 != nil && "GenerateCrossingEventFromChild1ToChild2, child2 parameter is nil");
   assert(theEvent != nil && "GenerateCrossingEventFromChild1ToChild2, theEvent parameter is nil");
   
   //ROOT does not have NotifyNonlinear/NotifyNonlinearVirtual.
   
   if (child1.fEventMask & kLeaveWindowMask)
      SendLeaveEvent(child1, theEvent, detail);
   
   if (!ancestor) {
      //From child1 to it's top-level view.
      if (child1.fParentView)
         SendLeaveEventClosedRange(child1.fParentView, (NSView<X11Window> *)[[child1 window] contentView], theEvent, detail);
      if (child2.fParentView)
         SendEnterEventClosedRange(child2.fParentView, (NSView<X11Window> *)[[child2 window] contentView], theEvent, detail);
   } else {
      if (child1.fParentView)
         SendLeaveEventRange(child1.fParentView, ancestor, theEvent, detail);
      if (child2.fParentView)
         SendEnterEventRange(child2.fParentView, ancestor, theEvent, detail);
   }

   if (child2.fEventMask & kEnterWindowMask)
      SendEnterEvent(child2, theEvent, detail);
}

}//Detail

//______________________________________________________________________________
EventTranslator::EventTranslator()
                     : fViewUnderPointer(nil),
                       fPointerGrab(kPGNoGrab),
                       fGrabEventMask(0),
                       fOwnerEvents(true),
                       fButtonGrabView(nil),
                       fKeyGrabView(nil),
                       fFocusView(nil)
                       
{
}

//______________________________________________________________________________
void EventTranslator::GenerateConfigureNotifyEvent(NSView<X11Window> *view, const NSRect &newFrame)
{
   assert(view != nil && "GenerateConfigureNotifyEvent, view parameter is nil");

   Event_t newEvent = {};
   newEvent.fWindow = view.fID;
   newEvent.fType = kConfigureNotify;         

   newEvent.fX = newFrame.origin.x;
   newEvent.fY = newFrame.origin.y;
   //fXRoot?
   //fYRoot?
   newEvent.fWidth = newFrame.size.width;
   newEvent.fHeight = newFrame.size.height;

   TGWindow *window = gClient->GetWindowById(view.fID);
   assert(window != 0 && "GenerateConfigureNotifyEvent, window was not found");   
   window->HandleEvent(&newEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateDestroyNotify(unsigned /*winID*/)
{
/*
   if (view.fEventMask & kStructureNotifyMask) {
      Event_t event = {};
      event.fHandle = view.fID;
      
      TGWindow *window = gClient->GetWindowById(view.fID);
      assert(window != 0 && "SendEnterEvent, window was not found");

   }*/
}

//______________________________________________________________________________
void EventTranslator::GenerateExposeEvent(NSView<X11Window> *view, const NSRect &exposedRect)
{
   assert(view != nil && "GenerateExposeEvent, view parameter is nil");
   
   Event_t exposeEvent = {};
   exposeEvent.fWindow = view.fID;
   exposeEvent.fType = kExpose;
   exposeEvent.fX = exposedRect.origin.x;
   exposeEvent.fY = exposedRect.origin.y;
   exposeEvent.fWidth = exposedRect.size.width;
   exposeEvent.fHeight = exposedRect.size.height;

   TGWindow *window = gClient->GetWindowById(view.fID);
   assert(window != 0 && "GenerateExposeEvent, window was not found");
   window->HandleEvent(&exposeEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEvent(NSView<X11Window> *view, NSEvent *theEvent)
{
   //View parameter can be nil (we exit any window).
   assert(theEvent != nil && "GenerateCrossingEvent, event parameter is nil");

   if (fPointerGrab == kPGNoGrab) {
      NSView *candidateView = [[[view window] contentView] hitTest : [theEvent locationInWindow]];
      const bool isROOTView = [candidateView isKindOfClass : [QuartzView class]] || [candidateView isKindOfClass : [ROOTOpenGLView class]];
      if (candidateView && !isROOTView) {
         NSLog(@"EventTranslator::GenerateCrossingEvent: error, hit test returned not a QuartzView!");
         candidateView = nil;
      }

      GenerateCrossingEvent((NSView<X11Window> *)candidateView, theEvent, kNotifyNormal);
   } else
      GenerateCrossingEventActiveGrab(view, theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEventActiveGrab(NSView<X11Window> *view, NSEvent *theEvent)
{
   assert(view != nil && "GenerateCrossingEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateCrossingEventActiveGrab, event parameter is nil");

   if (!fButtonGrabView)//implicit grab with 'root'?
      return;
      
   if (fOwnerEvents) {
      NSView<X11Window> *candidateView = nil;
      SortTopLevelWindows();
      QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent();
      if (topLevel) {
         const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
         candidateView = (NSView<X11Window> *)[[topLevel contentView] hitTest : mousePosition];
         if (candidateView)
            //Do propagation.
            candidateView = Detail::FindViewToPropagateEvent(candidateView, kEnterWindowMask | kLeaveWindowMask, fButtonGrabView, fGrabEventMask);
      }
      
      GenerateCrossingEvent(candidateView, theEvent, kNotifyNormal);
   } else {
      if (view == fButtonGrabView) {//We enter or leave grab view.
         const NSEventType type = [theEvent type];
         if (type == NSMouseEntered && (fButtonGrabView.fGrabButtonEventMask & kEnterWindowMask)) {
            if (fViewUnderPointer != fButtonGrabView) {//Can it be false???
               Detail::SendEnterEvent(fButtonGrabView, theEvent, kNotifyNormal);
               fViewUnderPointer = fButtonGrabView;
            }
         } 
         
         if (type == NSMouseExited && (fButtonGrabView.fGrabButtonEventMask & kEnterWindowMask)) {
            Detail::SendLeaveEvent(fButtonGrabView, theEvent, kNotifyNormal);
            //Who is now under pointer?
            fViewUnderPointer = nil;
         }
      }
   }
}

//______________________________________________________________________________
bool EventTranslator::HasPointerGrab()const
{
   return fPointerGrab != kPGNoGrab;
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEvent(NSView<X11Window> *view, NSEvent *theEvent, EXMagic detail)
{
   assert(theEvent != nil && "GenerateCrossingEvent, event parameter is nil");

   if (view == fViewUnderPointer) {
      //This can happen: tracking areas for stacked windows call
      //mouseExited even for overlapped views (so you have a bunch of mouseExited/mouseEntered
      //for one cursor move). In mouseEntered/mouseExited
      //I'm looking for the top level view under cursor and try to generate cross event
      //for this view only.
      return;
   }

   if (!fViewUnderPointer) {
      //We enter window "from the screen" - do not leave any window.
      //Send EnterNotify event.
      if (view)//Check, if order is OK.
         Detail::SendEnterEventClosedRange(view, (NSView<X11Window> *)[[view window] contentView], theEvent, detail);
   } else if (!view) {
      //We exit all views. Order must be OK here.
      Detail::SendLeaveEventClosedRange(fViewUnderPointer, (NSView<X11Window> *)[[fViewUnderPointer window] contentView], theEvent, detail);
   } else {
      NSView<X11Window> *ancestor = 0;
      Ancestry rel = FindRelation(fViewUnderPointer, view, &ancestor);
      if (rel == kAView1IsParent) {
         //Case 1.
         //From A to B.
         //_________________
         //| A              |
         //|   |---------|  |
         //|   |  B      |  |
         //|   |         |  |
         //|   |---------|  |
         //|                |
         //|________________|
         Detail::GenerateCrossingEventParentToChild(fViewUnderPointer, view, theEvent, detail);
      } else if (rel == kAView2IsParent) {
         //Case 2.
         //From A to B.
         //_________________
         //| B              |
         //|   |---------|  |
         //|   |  A      |  |
         //|   |         |  |
         //|   |---------|  |
         //|                |
         //|________________|   
         Detail::GenerateCrossingEventChildToParent(view, fViewUnderPointer, theEvent, detail);
      } else {
         //Case 3.
         //|--------------------------------|
         //| C   |------|      |-------|    |
         //|     | A    |      | B     |    |
         //|     |______|      |_______|    |
         //|________________________________|
         //Ancestor is either some view, or 'root' window.
         //The fourth case (different screens) is not implemented (and I do not know, if I want to implement it).
         Detail::GenerateCrossingEventFromChild1ToChild2(fViewUnderPointer, view, ancestor, theEvent, detail);
      }
   }
   
   fViewUnderPointer = view;
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEvent(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GeneratePointerMotionEvent, view parameter is nil");
   assert(theEvent != nil && "GeneratePointerMotionEvent, event parameter is nil");

   if (fPointerGrab == kPGNoGrab) {
      return GeneratePointerMotionEventNoGrab(eventView, theEvent);
   } else {
      return GeneratePointerMotionEventActiveGrab(eventView, theEvent);
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEvent(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn)
{
   assert(eventView != nil && "GenerateButtonPressEvent, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonpressEvent, event parameter is nil");
   
   if (fPointerGrab == kPGNoGrab)
      return GenerateButtonPressEventNoGrab(eventView, theEvent, btn);
   else
      return GenerateButtonPressEventActiveGrab(eventView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEvent(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn)
{
   assert(eventView != nil && "GenerateButtonReleaseEvent, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEvent, event parameter is nil");
   
   if (fPointerGrab == kPGNoGrab)
      return GenerateButtonReleaseEventNoGrab(eventView, theEvent, btn);
   else
      return GenerateButtonReleaseEventActiveGrab(eventView, theEvent, btn);
   
   
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyPressEvent(NSView<X11Window> *view, NSEvent *theEvent)
{
   assert(view != nil && "GenerateKeyPressEvent, view parameter is nil");
   (void)view;//TODO: change interface?
   assert(theEvent != nil && "GenerateKeyPressEvent, theEvent parameter is nil");
   
   if (![[theEvent charactersIgnoringModifiers] length])
      return;

   if (!fKeyGrabView && !fFocusView)
      return;
   
   !fKeyGrabView ? GenerateKeyPressEventNoGrab(theEvent) : 
                   GenerateKeyEventActiveGrab(theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyReleaseEvent(NSView<X11Window> *view, NSEvent *theEvent)
{
   assert(view != nil && "GenerateKeyReleaseEvent, view parameter is nil");
   (void)view;//TODO: change interface?
   assert(theEvent != nil && "GenerateKeyReleaseEvent, theEvent parameter is nil");

   if (![[theEvent charactersIgnoringModifiers] length])
      return;

   if (!fKeyGrabView && !fFocusView)
      return;
   
   !fKeyGrabView ? GenerateKeyReleaseEventNoGrab(theEvent) : 
                   GenerateKeyEventActiveGrab(theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateFocusChangeEvent(NSView<X11Window> *eventView)
{
   if (eventView == fFocusView)
      return;

   if (fFocusView && (fFocusView.fEventMask & kFocusChangeMask))
      Detail::SendFocusOutEvent(fFocusView, kNotifyNormal);

   if (eventView) {
      if (eventView.fEventMask & kFocusChangeMask)
         Detail::SendFocusInEvent(eventView, kNotifyNormal);

      fFocusView = eventView;
   } else
      fFocusView = nil;
}

//______________________________________________________________________________
void EventTranslator::SetPointerGrab(NSView<X11Window> *grabView, unsigned eventMask, bool ownerEvents)
{
   assert(grabView != nil && "SetPointerGrab, view parameter is nil");
   
   //Now some magic to receive mouse move events even outside any window.
   //if (eventMask & kPointerMotionMask)
   //   [[grabView window] setAcceptsMouseMovedEvents : YES];
   
   fButtonGrabView = grabView;
   fPointerGrab = kPGActiveGrab;
   fGrabEventMask = eventMask;
   fOwnerEvents = ownerEvents;
}

//______________________________________________________________________________
void EventTranslator::CancelPointerGrab()
{
   if (!fButtonGrabView)
      return;
      
   //[[fButtonGrabView window] setAcceptsMouseMovedEvents : NO];//Do not track mouse move events outside window anymore.
   
   fButtonGrabView = nil;
   fPointerGrab = kPGNoGrab;
   fGrabEventMask = 0;
   fOwnerEvents = true;
}

//______________________________________________________________________________
void EventTranslator::SetInputFocus(NSView<X11Window> *newFocusView)
{
   if (fFocusView && (fFocusView.fEventMask & kFocusChangeMask))
      Detail::SendFocusOutEvent(fFocusView, kNotifyNormal);

   if (newFocusView) {
      if (newFocusView.fEventMask & kFocusChangeMask)
         Detail::SendFocusInEvent(newFocusView, kNotifyNormal);

      fFocusView = newFocusView;
   } else
      fFocusView = nil;

}

//______________________________________________________________________________
unsigned EventTranslator::GetInputFocus()const
{
   if (fFocusView)
      return fFocusView.fID;
   
   return 0;
}

namespace {

//______________________________________________________________________________
void ClearPointerIfViewIsRelated(NSView<X11Window> *&view, Window_t winID) 
{
   NSView<X11Window> *v = view;
   if (v) {
      for (; v; v = v.fParentView) {
         if (v.fID == winID) {
            view = nil;
            break;
         }
      }
   }
}

}//unnamed namespace.

//______________________________________________________________________________
void EventTranslator::CheckUnmappedView(Window_t winID)
{
   //Window was unmapped, check, if it's the same window as the current grab,
   //or focus window, or key grabbing window and if so - do cleanup.
   
   //TODO: This is quite rough implementation - not sure, if this also has to 
   //generate some additional events.

   if (fButtonGrabView) {
      for (NSView<X11Window> *view = fButtonGrabView; view; view = view.fParentView) {
         if (view.fID == winID) {
            CancelPointerGrab();
            break;
         }
      }
   }
   
   ClearPointerIfViewIsRelated(fViewUnderPointer, winID);//TODO: send event to this view first?
   ClearPointerIfViewIsRelated(fFocusView, winID);//TODO: send event to this view first?
   ClearPointerIfViewIsRelated(fKeyGrabView, winID);//TODO: send event to this view first??
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   //Without grab, things are simple: find a view which accepts pointer motion event.

   assert(eventView != nil && "GeneratePointerMotionEventNoGrab, view parameter is nil");
   assert(theEvent != nil && "GeneratePointerMotionEventNoGrab, event parameter is nil");
   
   //Find a view on the top of stack:
   NSView<X11Window> *candidateView = (NSView<X11Window> *)[[[eventView window] contentView] hitTest : [theEvent locationInWindow]];
   if (candidateView) {
      //Do propagation.
      candidateView = Detail::FindViewToPropagateEvent(candidateView, kPointerMotionMask);
      if (candidateView)//We have such a view, send event to a corresponding ROOT's window.
         Detail::SendPointerMotionEvent(candidateView, theEvent);
   }
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEventActiveGrab(NSView<X11Window> * /*eventView*/, NSEvent *theEvent)
{
   //More complex case. Grab can be result of button press and set by SetPointerGrab.
   //In case of button press (this is either passive->active or implicit grab),
   //Cocoa has it's own grab, so view (and window) can be not under cursor (but still
   //it receives events). So I can not simple use eventView here.
   
   //TODO: change interface? - remove eventView parameter declaration.
   
   if (!fButtonGrabView)//Implicit grab when nobody has PressButtonMask
      return;
   
   //assert(eventView != nil && "GeneratePointerMotionEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GeneratePointerMotionEventActiveGrab, event parameter is nil");

   if (fOwnerEvents) {
      //Complex case, we have to correctly report event.
      SortTopLevelWindows();
      if (QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent()) {
         const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
         NSView<X11Window> *candidateView = (NSView<X11Window> *)[[topLevel contentView] hitTest : mousePosition];
         if (candidateView) {
            //Do propagation.
            candidateView = Detail::FindViewToPropagateEvent(candidateView, kPointerMotionMask, fButtonGrabView, fGrabEventMask);
            if (candidateView) {//We have such a view, send event to a corresponding ROOT's window.
               Detail::SendPointerMotionEvent(candidateView, theEvent);
            }
         }
      } else {
         //No such window - dispatch to the grab view.
         if (fGrabEventMask & kPointerMotionMask)
            Detail::SendPointerMotionEvent(fButtonGrabView, theEvent);
      }      
   } else {
      //Else: either implicit grab, or user requested grab with owner_grab == False.
      if (fGrabEventMask & kPointerMotionMask)
         Detail::SendPointerMotionEvent(fButtonGrabView, theEvent);
   }   
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEventNoGrab(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn)
{
   //Generate button press event when no pointer grab is active:
   //either find a window with a passive grab, or create an implicit
   //grab (to emulate X11's behavior).

   assert(view != nil && "GenerateButtonPressEventNoGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonPressEventNoGrab, event parameter is nil");

   FindButtonGrabView(view, theEvent, btn);
   //And now something badly defined. I tried X11 on mac and on linux, they do different things.
   //I'll do what was said in a spec and I do not care, if it's right or not, since there
   //is nothing 'right' in all this crap and mess. Since I'm activating grab,
   //before I send ButtonPress event, I'll send leave/enter notify events, if this is
   //required (previously entered view and current view are different).
   //If nothing was selected, on linux it looks like 'root' window
   //becomes a grab and all pointer events are discarded until ungrab.
   GenerateCrossingEvent(fButtonGrabView, theEvent, kNotifyGrab);
   
   if (fButtonGrabView)
      Detail::SendButtonPressEvent(fButtonGrabView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEventActiveGrab(NSView<X11Window> * /*view*/, NSEvent *theEvent, EMouseButton btn)
{
   //Generate button press event in the presence of activated pointer grab.

   //TODO: change interface? remove view parameter from declaration.

   //assert(view != nil && "GenerateButtonPressEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonPressEventActiveGrab, event parameter is nil");

   //I did not find in X11 spec. the case when I have two passive grabs on window A and window B,
   //say left button on A and right button on B. What should happen if I press left button in A, move to
   //B and press the right button? In my test programm on X11 (Ubuntu) I can see, that now they BOTH
   //are active grabs. I'm not going to implement this mess, unless I have a correct formal description.
   if (!fButtonGrabView)
      return;
      
   if (fOwnerEvents) {
      SortTopLevelWindows();
      if (QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent()) {
         const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
         NSView<X11Window> *candidateView = (NSView<X11Window> *)[[topLevel contentView] hitTest : mousePosition];
         if (candidateView) {
            //Do propagation.
            candidateView = Detail::FindViewToPropagateEvent(candidateView, kButtonPressMask, fButtonGrabView, fGrabEventMask);
            if (candidateView)//We have such a view, send event to a corresponding ROOT's window.
               Detail::SendButtonPressEvent(candidateView, theEvent, btn);
         }
      } else {
         if (fGrabEventMask & kButtonPressMask)
            Detail::SendButtonPressEvent(fButtonGrabView, theEvent, btn);
      }
   } else {
      if (fGrabEventMask & kButtonPressMask)
         Detail::SendButtonPressEvent(fButtonGrabView, theEvent, btn);
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn)
{
   //Generate button release event when there is no active pointer grab.

   assert(eventView != nil && "GenerateButtonReleaseEventNoGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEventNoGrabm event parameter is nil");
   
   if (NSView<X11Window> *candidateView = Detail::FindViewToPropagateEvent(eventView, kButtonPressMask))
      Detail::SendButtonReleaseEvent(candidateView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn)
{
   //Generate button release event in the presence of active grab (explicit pointer grab, activated passive grab or implicit grab).

   assert(eventView != nil && "GenerateButtonReleaseEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEventActiveGrab, event parameter is nil");
   
   const Util::NSStrongReference<NSView<X11Window> *> eventViewGuard(eventView);//What if view is deleted in the middle of this function?

   if (!fButtonGrabView) {
      if (fPointerGrab == kPGPassiveGrab || fPointerGrab == kPGImplicitGrab) {
         //'root' window was a grab window.
         fButtonGrabView = nil;
         fPointerGrab = kPGNoGrab;
         GenerateCrossingEvent(eventView, theEvent, kNotifyUngrab);
      }

      return;
   }
   
   bool needCrossingEvent = false;
   
   if (fOwnerEvents) {//X11: Either XGrabPointer with owner_events == True or passive grab (owner_events is always true)
      SortTopLevelWindows();
      if (QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent()) {
         const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
         NSView<X11Window> *candidateView = (NSView<X11Window> *)[[topLevel contentView] hitTest : mousePosition];
         if (candidateView) {
            candidateView = Detail::FindViewToPropagateEvent(candidateView, kButtonReleaseMask, fButtonGrabView, fGrabEventMask);
            if (candidateView) {//We have such a view, send event to a corresponding ROOT's window.
               needCrossingEvent = CancelImplicitOrPassiveGrab();
               Detail::SendButtonReleaseEvent(candidateView, theEvent, btn);
            }
         } else if (fGrabEventMask & kButtonReleaseMask) {
            NSView<X11Window> *grabView = fButtonGrabView;
            needCrossingEvent = CancelImplicitOrPassiveGrab();
            Detail::SendButtonReleaseEvent(grabView, theEvent, btn);
         }
      } else {//Report to the grab view, if it has a corresponding bit set.
         if (fGrabEventMask & kButtonReleaseMask) {
            NSView<X11Window> *grabView = fButtonGrabView;
            needCrossingEvent = CancelImplicitOrPassiveGrab();
            Detail::SendButtonReleaseEvent(grabView, theEvent, btn);
         }
      }
   } else {//Either implicit grab or XGrabPointer with owner_events == False.
      if (fGrabEventMask & kButtonReleaseMask) {
         NSView<X11Window> *grabView = fButtonGrabView;
         needCrossingEvent = CancelImplicitOrPassiveGrab();
         Detail::SendButtonReleaseEvent(grabView, theEvent, btn);   
      }
   }
   
   if (needCrossingEvent || CancelImplicitOrPassiveGrab())
      GenerateCrossingEvent(eventView, theEvent, kNotifyUngrab);
}

//______________________________________________________________________________
bool EventTranslator::CancelImplicitOrPassiveGrab()
{
   //Cancel the current grab, if it's implicit or passive button grab.

   if (fPointerGrab == kPGPassiveGrab || fPointerGrab == kPGImplicitGrab) {
      fButtonGrabView = nil;
      fPointerGrab = kPGNoGrab;
      return true;
   }
   
   return false;
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyPressEventNoGrab(NSEvent *theEvent)
{
   assert(theEvent != nil && "GenerateKeyPressEventNoGrab, theEvent parameter is nil");
   assert(fFocusView != nil && "GenerateKeyPressEventNoGrab, fFocusView is nil");

   FindKeyGrabView(fFocusView, theEvent);

   if (!fKeyGrabView) {
      NSView<X11Window> *candidateView = nil;

      if ((candidateView = FindViewUnderPointer())) {
         if (Detail::IsParent(fFocusView, candidateView)) {
            FindKeyGrabView(candidateView, theEvent);
         }
      }
      
      if (!fKeyGrabView) {
         if (candidateView && Detail::IsParent(fFocusView, candidateView)) {
            GenerateKeyEventForView(candidateView, theEvent);
         } else
            GenerateKeyEventForView(fFocusView, theEvent);
         return;
      }
   }
   
   GenerateKeyEventActiveGrab(theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyEventActiveGrab(NSEvent *theEvent)
{
   assert(theEvent != nil && "GenerateKeyEventActiveGrab, theEvent parameter is nil");
   assert(fKeyGrabView != nil && "GenerateKeyEventActiveGrab, theEvent parameter is nil");
   
   if (NSView<X11Window> *candidateView = FindViewUnderPointer()) {
      //Since owner_events is always true in ROOT ...
      GenerateKeyEventForView(candidateView, theEvent);
   } else {// else part for grab view??
      GenerateKeyEventForView(fKeyGrabView, theEvent);
   }
   
   if (theEvent.type == NSKeyUp && fKeyGrabView) {
      //Cancel grab?
      NSString *characters = [theEvent charactersIgnoringModifiers];
      assert(characters != nil && "GenerateKeyEventActiveGrab, [theEvent characters] returned nil");
      assert([characters length] > 0 && "GenerateKeyEventActiveGrab, characters is an empty string");

      if ([fKeyGrabView findPassiveKeyGrab : [characters characterAtIndex : 0]])
         fKeyGrabView = nil;//Cancel grab.
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyReleaseEventNoGrab(NSEvent *theEvent)
{
   assert(theEvent != nil && "GenerateKeyReleaseEventNoGrab, theEvent parameter is nil");
   
   NSView<X11Window> *candidateView = FindViewUnderPointer();

   if (candidateView && Detail::IsParent(fFocusView, candidateView))
      GenerateKeyEventForView(candidateView, theEvent);
   else 
      GenerateKeyEventForView(fFocusView, theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyEventForView(NSView<X11Window> *view, NSEvent *theEvent)
{
   //Generate key press event for a view without grab.
   assert(view != nil && "GenerateKeyEventForView, view parameter is nil");
   assert(theEvent != nil && "GenerateKeyEventForView, theEvent parameter is nil");
   assert(theEvent.type == NSKeyDown || theEvent.type == NSKeyUp && 
          "GenerateKeyEvenForView, event's type must be keydown or keyup");
   
   const Mask_t eventType = theEvent.type == NSKeyDown ? kKeyPressMask : kKeyReleaseMask;
   NSView<X11Window> *childView = nil;

   for (;;) {
      if (!view.isHidden && (view.fEventMask & eventType))
         break;
      if (!view.fParentView)
         return;
      //TODO: Also, check do not propagate mask here?
      childView = view.isHidden ? nil : view;
      view = view.fParentView;
   }
      
   NSPoint mousePosition = {};
   SortTopLevelWindows();
   if (QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent())
      mousePosition = [topLevel mouseLocationOutsideOfEventStream];

   if (eventType == kKeyPressMask)
      Detail::SendKeyPressEvent(view, childView, theEvent, mousePosition);
   else;
}

//______________________________________________________________________________
void EventTranslator::FindButtonGrabView(NSView<X11Window> *fromView, NSEvent *theEvent, EMouseButton btn)
{
   assert(fromView != nil && "FindButtonGrabView, view parameter is nil");
   assert(theEvent != nil && "FindButtonGrabView, event parameter is nil");

   const unsigned keyModifiers = Detail::GetKeyboardModifiersFromCocoaEvent(theEvent);
   
   NSView<X11Window> *grabView = 0;
   NSView<X11Window> *buttonPressView = 0;
   
   for (NSView<X11Window> *view = fromView; view != nil; view = view.fParentView) {
      //Top-first view to receive button press event.
      if (!buttonPressView && (view.fEventMask & kButtonPressMask))
         buttonPressView = view;

      //Bottom-first view with passive grab.
      if (view.fGrabButton == kAnyButton || view.fGrabButton == btn) {
         //Check modifiers.
         if (view.fGrabKeyModifiers == kAnyModifier || (view.fGrabKeyModifiers & keyModifiers))
            grabView = view;
      }
   }
   
   if (grabView) {
      fButtonGrabView = grabView;
      fPointerGrab = kPGPassiveGrab;
      fGrabEventMask = grabView.fGrabButtonEventMask;
      fOwnerEvents = grabView.fOwnerEvents;
   } else if (buttonPressView) {
      //This is implicit grab.
      fButtonGrabView = buttonPressView;
      fPointerGrab = kPGImplicitGrab;
      fGrabEventMask = buttonPressView.fEventMask;//?
      fOwnerEvents = false;
   } else {
      //Implicit grab with 'root' window?
      fButtonGrabView = nil;
      fPointerGrab = kPGImplicitGrab;
      fGrabEventMask = 0;
      fOwnerEvents = false;
   }
}

//______________________________________________________________________________
void EventTranslator::FindKeyGrabView(NSView<X11Window> *fromView, NSEvent *theEvent)
{
   assert(fromView != nil && "FindKeyGrabView, fromView parameter is nil");
   assert(theEvent != nil && "FindKeyGrabView, theEvent parameter is nil");

   NSString *characters = [theEvent charactersIgnoringModifiers];
   assert(characters != nil && "FindKeyGrabView, [theEvent characters] returned nil");
   assert([characters length] > 0 && "FindKeyGrabView, characters is an empty string");

   const NSUInteger modifiers = [theEvent modifierFlags];
   const unichar keyCode = [characters characterAtIndex : 0];

   for (NSView<X11Window> *v = fromView; v; v = v.fParentView) {
      if ([v findPassiveKeyGrab : keyCode modifiers : modifiers])
         fKeyGrabView = v;
   }
}

//______________________________________________________________________________
NSView<X11Window> *EventTranslator::FindViewUnderPointer()
{
   SortTopLevelWindows();
   if (QuartzWindow *topLevel = FindTopLevelWindowForMouseEvent()) {
      const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
      if (NSView<X11Window> *candidateView = (NSView<X11Window> *)[[topLevel contentView] hitTest : mousePosition])
         return candidateView;
   }
   
   return nil;
}

//______________________________________________________________________________
Ancestry EventTranslator::FindRelation(NSView<X11Window> *view1, NSView<X11Window> *view2, NSView<X11Window> **lca)
{
   assert(view1 != nil && "FindRelation, view1 parameter is nil");
   assert(view2 != nil && "FindRelation, view2 parameter is nil");
   assert(lca != 0 && "FindRelation, lca parameter is nil");
   
   if (Detail::IsParent(view1, view2)) 
      return kAView1IsParent;
   
   if (Detail::IsParent(view2, view1))
      return kAView2IsParent;
   
   //TODO: check if I can use [view1 ancestorSharedWithView : view2];
   return Detail::FindLowestCommonAncestor(view1, fBranch1, view2, fBranch2, lca);
}

//______________________________________________________________________________
void EventTranslator::SortTopLevelWindows()
{
   const ROOT::MacOSX::Util::AutoreleasePool pool;

   fWindowStack.clear();

   NSArray *orderedWindows = [NSApp orderedWindows];
   for (NSWindow *window in orderedWindows) {
      if (![window isKindOfClass : [QuartzWindow class]])
         continue;
      QuartzWindow *qw = (QuartzWindow *)window;
      if (qw.fMapState == kIsViewable)
         fWindowStack.push_back((QuartzWindow *)window);
   }
}

//______________________________________________________________________________
QuartzWindow *EventTranslator::FindTopLevelWindowForMouseEvent()
{
   if (!fWindowStack.size())
      return nil;

   std::vector<QuartzWindow *>::iterator iter = fWindowStack.begin(), endIt = fWindowStack.end();
   for (; iter != endIt; ++iter) {
      QuartzWindow *topLevel = *iter;
      const NSPoint mousePosition = [topLevel mouseLocationOutsideOfEventStream];
      const NSSize windowSize = topLevel.frame.size;
      if (mousePosition.x >= 0 && mousePosition.x <= windowSize.width && 
          mousePosition.y >= 0 && mousePosition.y <= windowSize.height)
         return topLevel;
   }
   
   return nil;
}

}//X11
}//MacOSX
}//ROOT
