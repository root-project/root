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

@interface FakeCrossingEvent : NSEvent {
   NSWindow *fQuartzWindow;
   NSPoint fLocationInWindow;
}

@end

@implementation FakeCrossingEvent

//______________________________________________________________________________
- (id) initWithWindow : (NSWindow *) window location : (NSPoint) location
{
   //Window should be always non-nil: we either enter some window, or exit some window.
   assert(window && "initWithWindow:location:, parameter 'window' is nil");

   if (self = [super init]) {
      fQuartzWindow = window;
      fLocationInWindow = location;
   }

   return self;
}

//______________________________________________________________________________
- (NSWindow *) window
{
   assert(fQuartzWindow && "window, fQuartzWindow is nil");
   return fQuartzWindow;
}

//______________________________________________________________________________
- (NSPoint) locationInWindow
{
   assert(fQuartzWindow != nil && "locationInWindow, fQuartzWindow is nil");
   return fLocationInWindow;
}

//______________________________________________________________________________
- (NSTimeInterval) timestamp
{
   //Hehe.
   return 0.;
}

@end


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
   assert(buf != 0 && "MapUnicharToKeySym, parameter 'buf' is null");

   //TODO: something really weird :)
   //read how XLookupString actually works? ;)

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
      buf[0] = key <= 0x7e ? key : 0;
      rootKeySym = iter->fSecond;
   } else {
      buf[0] = key;//????
      rootKeySym = key;
   }
}

//______________________________________________________________________________
Int_t MapKeySymToKeyCode(Int_t keySym)
{
   //Apart from special keys, ROOT has also ASCII symbols, they map directly to themselves.
   if (keySym >= 0x20 && keySym <= 0x7e)
      return keySym;

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

//______________________________________________________________________________
UInt_t GetKeyboardModifiers()
{
   const NSUInteger modifiers = [NSEvent modifierFlags];

   UInt_t rootModifiers = 0;
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
UInt_t GetModifiers()
{
   UInt_t rootModifiers = GetKeyboardModifiers();
   const NSUInteger buttons = [NSEvent pressedMouseButtons];
   if (buttons & 1)
      rootModifiers |= kButton1Mask;
   if (buttons & 2)
      rootModifiers |= kButton3Mask;
   if (buttons & (1 << 2))
      rootModifiers |= kButton2Mask;

   return rootModifiers;
}

namespace Detail {

#pragma mark - Several aux. functions to extract parameters from Cocoa events.

//______________________________________________________________________________
Time_t TimeForCocoaEvent(NSEvent *theEvent)
{
   //1. Event is not nil.
   assert(theEvent != nil && "TimeForCocoaEvent, parameter 'theEvent' is nil");

   return [theEvent timestamp] * 1000;//TODO: check this!
}

//______________________________________________________________________________
Event_t NewX11EventFromCocoaEvent(unsigned windowID, NSEvent *theEvent)
{
   //1. Event is not nil.

   assert(theEvent != nil && "NewX11EventFromCocoaEvent, parameter 'theEvent' is nil");

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

   assert(cocoaEvent != nil && "ConvertEventLocationToROOTXY, parameter 'cocoaEvent' is nil");
   assert(eventView != nil && "ConvertEventLocationToROOTXY, parameter 'eventView' is nil");
   assert(rootEvent != 0 && "ConvertEventLocationToROOTXY, parameter 'rootEvent' is null");

   //TODO: can [event window] be nil? (this can probably happen with mouse grabs).
   if (![cocoaEvent window])
      NSLog(@"Error in ConvertEventLocationToROOTXY, window property"
             " of event is nil, can not convert coordinates correctly");

   //Due to some reason, Apple has deprectated point conversion and requires to convert ... a rect.
   //Even more, on HiDPI point conversion produces wrong results and rect conversion works.

   const NSPoint screenPoint = ConvertPointFromBaseToScreen([cocoaEvent window], [cocoaEvent locationInWindow]);
   const NSPoint winPoint = ConvertPointFromScreenToBase(screenPoint, [eventView window]);
   const NSPoint viewPoint = [eventView convertPoint : winPoint fromView : nil];

   rootEvent->fX = viewPoint.x;
   rootEvent->fY = viewPoint.y;

   rootEvent->fXRoot = GlobalXCocoaToROOT(screenPoint.x);
   rootEvent->fYRoot = GlobalYCocoaToROOT(screenPoint.y);
}

//______________________________________________________________________________
unsigned GetKeyboardModifiersFromCocoaEvent(NSEvent *theEvent)
{
   assert(theEvent != nil && "GetKeyboardModifiersFromCocoaEvent, parameter 'event' is nil");

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
   assert(theEvent != nil && "GetModifiersFromCocoaEvent, parameter 'event' is nil");

   unsigned rootModifiers = GetKeyboardModifiersFromCocoaEvent(theEvent);
   const NSUInteger buttons = [NSEvent pressedMouseButtons];
   if (buttons & 1)
      rootModifiers |= kButton1Mask;
   if (buttons & 2)
      rootModifiers |= kButton3Mask;
   if (buttons & (1 << 2))
      rootModifiers |= kButton2Mask;

   return rootModifiers;
}

#pragma mark - Misc. aux. functions.

//______________________________________________________________________________
bool IsParent(NSView<X11Window>  *testParent, NSView<X11Window>  *testChild)
{
   assert(testParent != nil && "IsParent, parameter 'testParent' is nil");
   assert(testChild != nil && "IsParent, parameter 'testChild' is nil");

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
bool IsInBranch(NSView<X11Window> *parent, NSView<X11Window> *child, NSView<X11Window> *testView)
{
   assert(child != nil && "IsInBranch, parameter 'child' is nil");
   assert(testView != nil && "IsInBranch, parameter 'testView' is nil");

   if (testView == child || testView == parent)
      return true;

   for (NSView<X11Window> *current = child.fParentView; current != parent; current = current.fParentView) {
      if (current == testView)
         return true;
   }

   return false;
}

//Relation between two views.
enum Ancestry {
   kAView1IsParent,
   kAView2IsParent,
   kAHaveNonRootAncestor,
   kAAncestorIsRoot
};

//______________________________________________________________________________
Ancestry FindLowestCommonAncestor(NSView<X11Window> *view1, NSView<X11Window> *view2,
                                  NSView<X11Window> **lca)
{
   //Search for the lowest common ancestor.
   //View1 can not be parent of view2, view2 can not be parent of view1,
   //I do not check this condition here.

   assert(view1 != nil && "FindLowestCommonAncestor, parameter 'view1' is nil");
   assert(view2 != nil && "findLowestCommonAncestor, parameter 'view2' is nil");
   assert(lca != 0 && "FindLowestCommonAncestor, parameter 'lca' is null");

   if (!view1.fParentView)
      return kAAncestorIsRoot;

   if (!view2.fParentView)
      return kAAncestorIsRoot;

   NSView<X11Window> * const ancestor = (NSView<X11Window> *)[view1 ancestorSharedWithView : view2];

   if (ancestor) {
      *lca = ancestor;
      return kAHaveNonRootAncestor;
   }

   return kAAncestorIsRoot;
}

//______________________________________________________________________________
Ancestry FindRelation(NSView<X11Window> *view1, NSView<X11Window> *view2, NSView<X11Window> **lca)
{
   assert(view1 != nil && "FindRelation, view1 parameter is nil");
   assert(view2 != nil && "FindRelation, view2 parameter is nil");
   assert(lca != 0 && "FindRelation, lca parameter is nil");

   if (IsParent(view1, view2))
      return kAView1IsParent;

   if (IsParent(view2, view1))
      return kAView2IsParent;

   return FindLowestCommonAncestor(view1, view2, lca);
}

//______________________________________________________________________________
NSView<X11Window> *FindViewToPropagateEvent(NSView<X11Window> *viewFrom, Mask_t checkMask)
{
   //This function does not check passive grabs.
   assert(viewFrom != nil && "FindViewToPropagateEvent, parameter 'view' is nil");

   if (viewFrom.fEventMask & checkMask)
      return viewFrom;

   for (viewFrom = viewFrom.fParentView; viewFrom; viewFrom = viewFrom.fParentView) {
      if (viewFrom.fEventMask & checkMask)
         return viewFrom;
   }

   return nil;
}

//______________________________________________________________________________
NSView<X11Window> *FindViewToPropagateEvent(NSView<X11Window> *viewFrom, Mask_t checkMask,
                                            NSView<X11Window> *grabView, Mask_t grabMask)
{
   //This function is called when we have a grab and owner_events == true,
   //in this case the grab view itself (and its grab mask) is checked
   //at the end (if no view was found before). Grab view can be in a hierarchy
   //for a 'viewFrom' view and can have matching fEventMask.

   assert(viewFrom != nil && "FindViewToPropagateEvent, parameter 'view' is nil");

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

#pragma mark - Aux. 'low-level' functions to generate events and call HandleEvent for a root window.

//______________________________________________________________________________
void SendEnterEvent(EventQueue_t &queue, NSView<X11Window> *view, NSEvent *theEvent,
                    EXMagic detail)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //This view must receive enter notify, I do not check it here.

   assert(view != nil && "SendEnterEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendEnterEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendEnterEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
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

   //Enqueue event again.
   queue.push_back(enterEvent);
}

//______________________________________________________________________________
void SendLeaveEvent(EventQueue_t &queue, NSView<X11Window> *view, NSEvent *theEvent,
                    EXMagic detail)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //This window should receive leave event, I do not check it here.

   assert(view != nil && "SendLeaveEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendLeaveEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendLeaveEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
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
   //Enqueue event for ROOT.
   queue.push_back(leaveEvent);
}

//______________________________________________________________________________
void SendPointerMotionEvent(EventQueue_t &queue, NSView<X11Window> *view, NSEvent *theEvent)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View receives pointer motion events, I do not check this condition here.

   assert(view != nil && "SendPointerMotionEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendPointerMotionEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendPointerMotionEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendPointerMotionEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t motionEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   motionEvent.fType = kMotionNotify;
   motionEvent.fState = GetModifiersFromCocoaEvent(theEvent);

   ConvertEventLocationToROOTXY(theEvent, view, &motionEvent);
   //Enqueue event for ROOT.
   queue.push_back(motionEvent);
}

//______________________________________________________________________________
void SendButtonPressEvent(EventQueue_t &queue, NSView<X11Window> *view, NSEvent *theEvent,
                          EMouseButton btn)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View receives this event (either grab or select input)
   //   - I do not check this condition here.

   assert(view != nil && "SendButtonPressEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendButtonPressEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendButtonPressEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
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

   //Enqueue event for ROOT.
   queue.push_back(pressEvent);
}

//______________________________________________________________________________
void SendButtonReleaseEvent(EventQueue_t &queue, NSView<X11Window> *view, NSEvent *theEvent,
                            EMouseButton btn)
{
   //1. Parameters are valid.
   //2. view.fID is valid.
   //3. A window for view.fID exists.
   //View must button release events, I do not check this here.

   assert(view != nil && "SendButtonReleaseEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendButtonReleaseEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendButtonReleaseEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
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
   //Enqueue for ROOT.
   queue.push_back(releaseEvent);
}

//______________________________________________________________________________
void SendKeyPressEvent(EventQueue_t &queue, NSView<X11Window> *view, NSView<X11Window> *childView,
                       NSEvent *theEvent, NSPoint windowPoint)
{
   assert(view != nil && "SendKeyPressEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendKeyPressEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendKeyPressEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendKeyPressEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t keyPressEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   keyPressEvent.fType = kGKeyPress;
   keyPressEvent.fState = GetKeyboardModifiersFromCocoaEvent(theEvent);

   NSString * const characters = [theEvent charactersIgnoringModifiers];
   assert(characters != nil && "SendKeyPressEvent, [theEvent characters] returned nil");
   assert([characters length] > 0 && "SendKeyPressEvent, characters is an empty string");

   keyPressEvent.fCode = [characters characterAtIndex : 0];

   //convertPointFromBase is deprecated.
   //const NSPoint viewPoint = [view convertPointFromBase : windowPoint];
   const NSPoint viewPoint = [view convertPoint : windowPoint fromView : nil];

   //Coords.
   keyPressEvent.fX = viewPoint.x;
   keyPressEvent.fY = viewPoint.y;
   const NSPoint screenPoint = TranslateToScreen(view, viewPoint);
   keyPressEvent.fXRoot = screenPoint.x;
   keyPressEvent.fYRoot = screenPoint.y;
   //Subwindow.
   if (childView)
      keyPressEvent.fUser[0] = childView.fID;

   //Enqueue for ROOT.
   queue.push_back(keyPressEvent);
}

//______________________________________________________________________________
void SendKeyReleaseEvent(EventQueue_t &queue, NSView<X11Window> *view, NSView<X11Window> *childView,
                         NSEvent *theEvent, NSPoint windowPoint)
{
   assert(view != nil && "SendKeyReleaseEvent, parameter 'view' is nil");
   assert(theEvent != nil && "SendKeyReleaseEvent, parameter 'event' is nil");
   assert(view.fID != 0 && "SendKeyReleaseEvent, view.fID is 0");

   TGWindow * const window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendKeyPressEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t keyReleaseEvent = NewX11EventFromCocoaEvent(view.fID, theEvent);
   keyReleaseEvent.fType = kKeyRelease;

   keyReleaseEvent.fState = GetKeyboardModifiersFromCocoaEvent(theEvent);

   NSString * const characters = [theEvent charactersIgnoringModifiers];
   assert(characters != nil && "SendKeyReleaseEvent, [theEvent characters] returned nil");
   assert([characters length] > 0 && "SendKeyReleaseEvent, characters is an empty string");
   keyReleaseEvent.fCode = [characters characterAtIndex : 0];

   //Coords.
   const NSPoint viewPoint = [view convertPoint : windowPoint fromView : nil];
   keyReleaseEvent.fX = viewPoint.x;
   keyReleaseEvent.fY = viewPoint.y;
   const NSPoint screenPoint = TranslateToScreen(view, viewPoint);

   keyReleaseEvent.fXRoot = screenPoint.x;
   keyReleaseEvent.fYRoot = screenPoint.y;

   //Subwindow.
   if (childView)
      keyReleaseEvent.fUser[0] = childView.fID;

   //Enqueue for ROOT.
   queue.push_back(keyReleaseEvent);
}


//______________________________________________________________________________
void SendFocusInEvent(EventQueue_t &queue, NSView<X11Window> *view, EXMagic mode)
{
   assert(view != nil && "SendFocusInEvent, parameter 'view' is nil");
   //
   TGWindow * const window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendFocusInEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t focusInEvent = {};
   focusInEvent.fWindow = view.fID;
   focusInEvent.fType = kFocusIn;
   focusInEvent.fCode = mode;

   queue.push_back(focusInEvent);
}

//______________________________________________________________________________
void SendFocusOutEvent(EventQueue_t &queue, NSView<X11Window> *view, EXMagic mode)
{
   assert(view != nil && "SendFocusOutEvent, parameter 'view' is nil");
   //
   TGWindow * const window = gClient->GetWindowById(view.fID);
   if (!window) {
#ifdef DEBUG_ROOT_COCOA
      NSLog(@"SendFocusOutEvent, ROOT's widget %u was not found", view.fID);
#endif
      return;
   }

   Event_t focusOutEvent = {};
   focusOutEvent.fWindow = view.fID;
   focusOutEvent.fType = kFocusOut;
   focusOutEvent.fCode = mode;//code mode :)

   queue.push_back(focusOutEvent);
}

#pragma mark - Aux. functions to send events to view's branch.

//______________________________________________________________________________
void SendEnterEventRange(EventQueue_t &queue, NSView<X11Window> *from, NSView<X11Window> *to,
                         NSEvent *theEvent, EXMagic mode)
{
   //[from, to) - legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendEnterEventRange, 'from' parameter is nil");
   assert(to != nil && "SendEnterEventRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendEnterEventRange, event parameter is nil");

   while (from != to) {
      if ([from acceptsCrossingEvents : kEnterWindowMask])
         SendEnterEvent(queue, from, theEvent, mode);
      from = from.fParentView;
   }
}

//______________________________________________________________________________
void SendEnterEventClosedRange(EventQueue_t &queue, NSView<X11Window> *from, NSView<X11Window> *to,
                               NSEvent *theEvent, EXMagic mode)
{
   //[from, to] - inclusive, legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendEnterEventClosedRange, 'from' parameter is nil");
   assert(to != nil && "SendEnterEventClosedRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendEnterEventClosedRange, event parameter is nil");

   SendEnterEventRange(queue, from, to, theEvent, mode);
   if ([to acceptsCrossingEvents : kEnterWindowMask])
      SendEnterEvent(queue, to, theEvent, mode);
}

//______________________________________________________________________________
void SendLeaveEventRange(EventQueue_t &queue, NSView<X11Window> *from, NSView<X11Window> *to,
                         NSEvent *theEvent, EXMagic mode)
{
   //[from, to) - legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendLeaveEventRange, 'from' parameter is nil");
   assert(to != nil && "SendLeaveEventRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendLeaveEventRange, event parameter is nil");

   while (from != to) {
      if ([from acceptsCrossingEvents : kLeaveWindowMask])
         SendLeaveEvent(queue, from, theEvent, mode);
      from = from.fParentView;
   }
}

//______________________________________________________________________________
void SendLeaveEventClosedRange(EventQueue_t &queue, NSView<X11Window> *from, NSView<X11Window> *to,
                               NSEvent *theEvent, EXMagic mode)
{
   //[from, to] - inclusive, legal range, 'to' must be ancestor for 'from'.
   assert(from != nil && "SendLeaveEventClosedRange, 'from' parameter is nil");
   assert(to != nil && "SendLeaveEventClosedRange, 'to' parameter is nil");
   assert(theEvent != nil && "SendLeaveEventClosedRange, event parameter is nil");

   SendLeaveEventRange(queue, from, to, theEvent, mode);
   if ([to acceptsCrossingEvents : kLeaveWindowMask])
      SendLeaveEvent(queue, to, theEvent, mode);
}

#pragma mark - Top-level crossing event generators.

//When passing parent and child view, parent view always
//precedes the child, even if function's name is GenerateCrossingEventChildToParent.

//______________________________________________________________________________
void GenerateCrossingEventChildToParent(EventQueue_t &queue, NSView<X11Window> *parent, NSView<X11Window> *child,
                                        NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and A is an inferior of B.
   //Generate LeaveNotify on A (with detail NotifyAncestor).
   //Generate LeaveNotify for every window between A and B, exclusive (with detail NotifyVirtual)
   //Generate EnterNotify for B with detail NotifyInferior.

   //ROOT does not have NotifyAncestor/NotifyInferior.

   assert(parent != nil && "GenerateCrossingEventChildToParent, parameter 'parent' is nil");
   assert(child != nil && "GenerateCrossingEventChildToParent, parameter 'child' is nil");
   assert(theEvent != nil && "GenerateCrossingEventChildToParent, parameter 'event' is nil");
   assert(child.fParentView != nil &&
          "GenerateCrossingEventChildToParent, parameter 'child' must have QuartzView* parent");

   //acceptsCrossingEvents will check grab event mask also, if view is a grab and if
   //owner_events == true.
   if ([child acceptsCrossingEvents : kLeaveWindowMask])
      SendLeaveEvent(queue, child, theEvent, detail);

   //Leave event to a branch [child.fParentView, parent)
   SendLeaveEventRange(queue, child.fParentView, parent, theEvent, detail);

   //Enter event for the parent view.
   if ([parent acceptsCrossingEvents : kEnterWindowMask])
      SendEnterEvent(queue, parent, theEvent, detail);
}

//______________________________________________________________________________
void GenerateCrossingEventParentToChild(EventQueue_t &queue, NSView<X11Window> *parent, NSView<X11Window> *child,
                                        NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and B is an inferior of A.
   //Generate LeaveNotify event for A, detail == NotifyInferior.
   //Generate EnterNotify for each window between window A and window B, exclusive,
   //    detail == NotifyVirtual (no such entity in ROOT).
   //Generate EnterNotify on window B, detail == NotifyAncestor.

   //ROOT does not have NotifyInferior/NotifyAncestor.

   assert(parent != nil && "GenerateCrossingEventParentToChild, parameter 'parent' is nil");
   assert(child != nil && "GenerateCrossingEventParentToChild, parameter 'child' is nil");
   assert(theEvent != nil && "GenerateCrossingEventParentToChild, parameter 'event' is nil");
   assert(child.fParentView != nil &&
          "GenerateCrossingEventParentToChild, parameter 'child' must have QuartzView* parent");

   //If view is a grab and owner_events == true,
   //acceptsCrossingEvents will check the grab event mask also.
   if ([parent acceptsCrossingEvents : kLeaveWindowMask])
      SendLeaveEvent(queue, parent, theEvent, detail);

   //Enter event for [child.fParentView, parent) - order is reversed, but it does not really matter.
   SendEnterEventRange(queue, child.fParentView, parent, theEvent, detail);

   //Enter event for the child view.
   if ([child acceptsCrossingEvents : kEnterWindowMask])
      SendEnterEvent(queue, child, theEvent, detail);
}

//______________________________________________________________________________
void GenerateCrossingEventFromChild1ToChild2(EventQueue_t &queue, NSView<X11Window> *child1, NSView<X11Window> *child2,
                                             NSView<X11Window> *ancestor, NSEvent *theEvent, EXMagic detail)
{
   //Pointer moves from window A to window B and window C is their lowest common ancestor.
   //Generate LeaveNotify for window A with detail == NotifyNonlinear.
   //Generate LeaveNotify for each window between A and C, exclusive, with detail == NotifyNonlinearVirtual
   //Generate EnterNotify (detail == NotifyNonlinearVirtual) for each window between C and B, exclusive
   //Generate EnterNotify for window B, with detail == NotifyNonlinear.

   assert(child1 != nil && "GenerateCrossingEventFromChild1ToChild2, parameter 'child1' is nil");
   assert(child2 != nil && "GenerateCrossingEventFromChild1ToChild2, child2 parameter is nil");
   assert(theEvent != nil && "GenerateCrossingEventFromChild1ToChild2, theEvent parameter is nil");

   //ROOT does not have NotifyNonlinear/NotifyNonlinearVirtual.

   //acceptsCrossingEvents also checks grab event mask, if this view has a grab
   //and owner_events == true.
   if ([child1 acceptsCrossingEvents : kLeaveWindowMask])
      SendLeaveEvent(queue, child1, theEvent, detail);

   if (!ancestor) {
      if (child1.fParentView)//Leave [child1.fParentView contentView]
         SendLeaveEventClosedRange(queue, child1.fParentView,
                                   (NSView<X11Window> *)[[child1 window] contentView], theEvent, detail);
      if (child2.fParentView)//Enter [child2.fParentView contentView] - order is reversed.
         SendEnterEventClosedRange(queue, child2.fParentView,
                                   (NSView<X11Window> *)[[child2 window] contentView], theEvent, detail);
   } else {
      if (child1.fParentView)//Leave [child1.fParentView ancestor)
         SendLeaveEventRange(queue, child1.fParentView, ancestor, theEvent, detail);
      if (child2.fParentView)//Enter [child2.fParentView, ancestor) - order reversed.
         SendEnterEventRange(queue, child2.fParentView, ancestor, theEvent, detail);
   }

   if ([child2 acceptsCrossingEvents : kEnterWindowMask])
      SendEnterEvent(queue, child2, theEvent, detail);
}


//______________________________________________________________________________
void GenerateCrossingEvents(EventQueue_t &queue, NSView<X11Window> *fromView, NSView<X11Window> *toView,
                            NSEvent *theEvent, EXMagic detail)
{
   //Pointer moved from 'fromView' to 'toView'.
   //Check their relationship and generate leave/enter notify events.

   assert(theEvent != nil && "GenerateCrossingEvent, event parameter is nil");

   if (fromView == toView) {
      //This can happen: tracking areas for stacked windows call
      //mouseExited even for overlapped views (so you have a bunch of mouseExited/mouseEntered
      //for one cursor move). In mouseEntered/mouseExited
      //I'm looking for the top level view under cursor and try to generate cross event
      //for this view only.
      return;
   }

   if (!fromView) {
      //We enter window "from the screen" - do not leave any window.
      //Send EnterNotify event.
      //Send enter notify event to a branch [toView contentView], order of
      //views is reversed, but no GUI actually depends on this.
      if (toView)
         SendEnterEventClosedRange(queue, toView, (NSView<X11Window> *)[[toView window] contentView],
                                   theEvent, detail);
   } else if (!toView) {
      //We exit all views. Order is correct here.
      SendLeaveEventClosedRange(queue, fromView, (NSView<X11Window> *)[[fromView window] contentView],
                                theEvent, detail);
   } else {
      NSView<X11Window> *ancestor = 0;
      const Ancestry rel = FindRelation(fromView, toView, &ancestor);
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
         GenerateCrossingEventParentToChild(queue, fromView, toView, theEvent, detail);
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
         GenerateCrossingEventChildToParent(queue, toView, fromView, theEvent, detail);
      } else {
         //Case 3.
         //|--------------------------------|
         //| C   |------|      |-------|    |
         //|     | A    |      | B     |    |
         //|     |______|      |_______|    |
         //|________________________________|
         //Ancestor is either some view, or 'root' window.
         //The fourth case (different screens) is not implemented (and I do not know, if I want to implement it).
         GenerateCrossingEventFromChild1ToChild2(queue, fromView, toView, ancestor, theEvent, detail);
      }
   }
}

//______________________________________________________________________________
void GenerateCrossingEventForGrabView(EventQueue_t &queue, NSView<X11Window> *fromView, NSView<X11Window> *toView,
                                      NSView<X11Window> *grabView, Mask_t grabEventMask, NSEvent *theEvent)
{
   //When owner events == false, only grab view receives enter/leave notify events.

   //Send enter/leave event to a grab view.
   assert(theEvent != nil && "GenerateCrossingEventForGrabView, parameter 'event' is nil");
   assert(grabView != nil && "GenerateCrossingEventForGrabView, parameter 'grabView' is nil");
   assert((fromView != nil || toView != nil) &&
          "GenerateCrossingEventForGrabView, both 'toView' and 'fromView' parameters are nil");

   if (fromView == toView)//No crossing at all?
      return;

   const bool wantsEnter = grabEventMask & kEnterWindowMask;
   const bool wantsLeave = grabEventMask & kLeaveWindowMask;

   if (fromView == grabView && wantsLeave)
      return SendLeaveEvent(queue, grabView, theEvent, kNotifyNormal);

   if (toView == grabView && wantsEnter)
      return SendEnterEvent(queue, grabView, theEvent, kNotifyNormal);

   if (!fromView) {
      //We enter window "from the screen" - do not leave any window.
      //Send EnterNotify event to the grab view, if it's "in the branch".
      if (wantsEnter && IsParent(grabView, toView))
         SendEnterEvent(queue, grabView, theEvent, kNotifyNormal);
   } else if (!toView) {
      //We exit all views..
      if (wantsLeave && IsParent(grabView, fromView))
         SendLeaveEvent(queue, grabView, theEvent, kNotifyNormal);
   } else {
      NSView<X11Window> *ancestor = 0;
      FindRelation(fromView, toView, &ancestor);

      if (IsInBranch(nil, fromView, grabView)) {
         if (wantsLeave)
            SendLeaveEvent(queue, grabView, theEvent, kNotifyNormal);
      } else if (IsInBranch(nil, toView, grabView)) {
         if (wantsEnter)
            SendEnterEvent(queue, grabView, theEvent, kNotifyNormal);
      }
   }
}

}//Detail

//______________________________________________________________________________
EventTranslator::EventTranslator()
                     : fViewUnderPointer(nil),
                       fPointerGrabType(kPGNoGrab),
                       fGrabEventMask(0),
                       fOwnerEvents(true),
                       fButtonGrabView(nil),
                       fKeyGrabView(nil),
                       fFocusView(nil),
                       fImplicitGrabButton(kAnyButton)

{
}

//______________________________________________________________________________
void EventTranslator::GenerateConfigureNotifyEvent(NSView<X11Window> *view, const NSRect &newFrame)
{
   assert(view != nil && "GenerateConfigureNotifyEvent, parameter 'view' is nil");

   Event_t newEvent = {};
   newEvent.fWindow = view.fID;
   newEvent.fType = kConfigureNotify;

   newEvent.fX = newFrame.origin.x;
   newEvent.fY = newFrame.origin.y;
   //fXRoot?
   //fYRoot?
   newEvent.fWidth = newFrame.size.width;
   newEvent.fHeight = newFrame.size.height;

   TGWindow * const window = gClient->GetWindowById(view.fID);
   assert(window != 0 && "GenerateConfigureNotifyEvent, window was not found");
   window->HandleEvent(&newEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateDestroyNotify(unsigned /*winID*/)
{
   //Noop.
}

//______________________________________________________________________________
void EventTranslator::GenerateExposeEvent(NSView<X11Window> *view, const NSRect &exposedRect)
{
   assert(view != nil && "GenerateExposeEvent, parameter 'view' is nil");

   Event_t exposeEvent = {};
   exposeEvent.fWindow = view.fID;
   exposeEvent.fType = kExpose;
   exposeEvent.fX = exposedRect.origin.x;
   exposeEvent.fY = exposedRect.origin.y;
   exposeEvent.fWidth = exposedRect.size.width;
   exposeEvent.fHeight = exposedRect.size.height;

   TGWindow * const window = gClient->GetWindowById(view.fID);
   assert(window != 0 && "GenerateExposeEvent, window was not found");
   window->HandleEvent(&exposeEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEvent(NSEvent *theEvent)
{
   //View parameter can be nil.
   //TODO: change interface, it looks like I do not need the 'view' parameter.
   assert(theEvent != nil && "GenerateCrossingEvent, parameter 'event' is nil");

   fPointerGrabType == kPGNoGrab ? GenerateCrossingEventNoGrab(theEvent) :
                                   GenerateCrossingEventActiveGrab(theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEventNoGrab(NSEvent *theEvent)
{
   assert(theEvent && "GenerateCrossingEventNoGrab, parameter 'theEvent' is nil");

   NSView<X11Window> * const candidateView = FindViewForPointerEvent(theEvent);
   //We moved from fViewUnderPointer (leave event) to candidateView (enter event).
   Detail::GenerateCrossingEvents(fEventQueue, fViewUnderPointer, candidateView, theEvent, kNotifyNormal);
   fViewUnderPointer = candidateView;
}

//______________________________________________________________________________
void EventTranslator::GenerateCrossingEventActiveGrab(NSEvent *theEvent)
{
   assert(theEvent != nil && "GenerateCrossingEventActiveGrab, parameter 'theEvent' is nil");

   NSView<X11Window> * const candidateView = FindViewForPointerEvent(theEvent);

   if (fOwnerEvents) {
      //Either passive grab (which was activated) or active grab set by TGCocoa::GrabPointer with
      //owner_events == true. This works the same way as nograb case, except not only fEventMask
      //is checked, but for grab view (if it's boundary was crossed) either it's passive grab mask
      //or active is also checked.
      Detail::GenerateCrossingEvents(fEventQueue, fViewUnderPointer, candidateView,
                                     theEvent, kNotifyNormal);
   } else if (fButtonGrabView && (fViewUnderPointer || candidateView)) {
      //Either implicit grab or GrabPointer with owner_events == false,
      //only grab view can receive enter/leave notify events. Only
      //grab event mask is checked, not view's own event mask.
      Detail::GenerateCrossingEventForGrabView(fEventQueue, fViewUnderPointer, candidateView,
                                               fButtonGrabView, fGrabEventMask, theEvent);
   }

   fViewUnderPointer = candidateView;
}

//______________________________________________________________________________
bool EventTranslator::HasPointerGrab()const
{
   return fPointerGrabType != kPGNoGrab;
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEvent(NSEvent *theEvent)
{
   assert(theEvent != nil && "GeneratePointerMotionEvent, parameter 'theEvent' is nil");



   if (fPointerGrabType == kPGNoGrab)
      return GeneratePointerMotionEventNoGrab(theEvent);
   else
      return GeneratePointerMotionEventActiveGrab(theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEvent(NSView<X11Window> *eventView, NSEvent *theEvent,
                                               EMouseButton btn)
{
   assert(eventView != nil && "GenerateButtonPressEvent, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateButtonpressEvent, parameter 'theEvent' is nil");

   if (fPointerGrabType == kPGNoGrab)
      return GenerateButtonPressEventNoGrab(eventView, theEvent, btn);
   else
      return GenerateButtonPressEventActiveGrab(eventView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEvent(NSView<X11Window> *eventView, NSEvent *theEvent,
                                                 EMouseButton btn)
{
   assert(eventView != nil && "GenerateButtonReleaseEvent, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEvent, parameter 'theEvent' is nil");

   if (fPointerGrabType == kPGNoGrab)
      return GenerateButtonReleaseEventNoGrab(eventView, theEvent, btn);
   else
      return GenerateButtonReleaseEventActiveGrab(eventView, theEvent, btn);


}

//______________________________________________________________________________
void EventTranslator::GenerateKeyPressEvent(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GenerateKeyPressEvent, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateKeyPressEvent, parameter 'theEvent' is nil");

   if (![[theEvent charactersIgnoringModifiers] length])
      return;

   if (!fFocusView)
      return;

   !fKeyGrabView ? GenerateKeyPressEventNoGrab(eventView, theEvent) :
                   GenerateKeyEventActiveGrab(eventView, theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyReleaseEvent(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GenerateKeyReleaseEvent, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateKeyReleaseEvent, parameter 'theEvent' is nil");

   if (![[theEvent charactersIgnoringModifiers] length])
      return;

   if (!fFocusView)
      return;

   !fKeyGrabView ? GenerateKeyReleaseEventNoGrab(eventView, theEvent) :
                   //GenerateKeyEventActiveGrab(eventView, theEvent);
                   GenerateKeyEventForView(fKeyGrabView, theEvent);

   //Oh, only God forgives.
   fKeyGrabView = nil;
}

//______________________________________________________________________________
void EventTranslator::GenerateFocusChangeEvent(NSView<X11Window> *eventView)
{
   if (eventView == fFocusView)
      return;

   if (fFocusView && (fFocusView.fEventMask & kFocusChangeMask))
      Detail::SendFocusOutEvent(fEventQueue, fFocusView, kNotifyNormal);

   if (eventView) {
      if (eventView.fEventMask & kFocusChangeMask)
         Detail::SendFocusInEvent(fEventQueue, eventView, kNotifyNormal);

      fFocusView = eventView;
   } else
      fFocusView = nil;
}

//______________________________________________________________________________
void EventTranslator::SetPointerGrab(NSView<X11Window> *grabView, unsigned eventMask, bool ownerEvents)
{
   assert(grabView != nil && "SetPointerGrab, parameter 'grabView' is nil");

   if (fButtonGrabView) {
      //This can happen with X11, does this happen with ROOT's GUI?
      //Hm, should I send leave notify to the previous grab???
      //TODO: check this!
      [fButtonGrabView cancelGrab];
   }

   //There is no kNoButton, unfortunately (but there is additional check on
   //grab type).
   fImplicitGrabButton = kAnyButton;

   //
   fButtonGrabView = grabView;
   fPointerGrabType = kPGActiveGrab;
   fGrabEventMask = eventMask;
   fOwnerEvents = ownerEvents;

   //Generate sequence of crossing events - as if pointer
   //"jumps" to the grab view.

   if (grabView != fViewUnderPointer) {
      const NSPoint location = [[grabView window] mouseLocationOutsideOfEventStream];
      const Util::NSScopeGuard<FakeCrossingEvent> event([[FakeCrossingEvent alloc] initWithWindow : [grabView window]
                                                        location : location]);
      if (!event.Get()) {
         //Hehe, if this happend, is it still possible to log????
         NSLog(@"EventTranslator::SetPointerGrab, crossing event initialization failed");
         return;
      }

      Detail::GenerateCrossingEvents(fEventQueue, fViewUnderPointer, grabView, event.Get(), kNotifyGrab);//Uffffff, done!
   }

   //Activate the current grab now.
   [fButtonGrabView activateGrab : eventMask ownerEvents : fOwnerEvents];
}

//______________________________________________________________________________
void EventTranslator::CancelPointerGrab()
{
   if (fButtonGrabView)
      //Cancel grab (active, passive, implicit).
      [fButtonGrabView cancelGrab];

   //We generate sequence of leave/enter notify events (if any) as if we jumped from the grab view to the pointer view.

   if (NSView<X11Window> * const candidateView = FindViewUnderPointer()) {
      const NSPoint location = [[candidateView window] mouseLocationOutsideOfEventStream];
      const Util::NSScopeGuard<FakeCrossingEvent> event([[FakeCrossingEvent alloc] initWithWindow : [candidateView window]
                                                        location : location ]);

      if (!event.Get()) {
         //Hehe, if this happend, is it still possible to log????
         NSLog(@"EventTranslator::CancelPointerGrab, crossing event initialization failed");
         return;
      }

      Detail::GenerateCrossingEvents(fEventQueue, fButtonGrabView, candidateView, event.Get(), kNotifyUngrab);
      //
      fViewUnderPointer = candidateView;
   } else if (fButtonGrabView) {
      //convertScreenToBase is deprecated.
      //const NSPoint location = [[fButtonGrabView window] convertScreenToBase : [NSEvent mouseLocation]];
      const NSPoint location = ConvertPointFromScreenToBase([NSEvent mouseLocation], [fButtonGrabView window]);

      const Util::NSScopeGuard<FakeCrossingEvent> event([[FakeCrossingEvent alloc] initWithWindow : [fButtonGrabView window]
                                                         location : location ]);

      if (!event.Get()) {
         //Hehe, if this happend, is it still possible to log????
         NSLog(@"EventTranslator::CancelPointerGrab, crossing event initialization failed");
         fViewUnderPointer = nil;
         return;
      }

      Detail::GenerateCrossingEvents(fEventQueue, fButtonGrabView, nil, event.Get(), kNotifyUngrab);//Ufff, done!!!
      //
      fViewUnderPointer = nil;
   }


   fImplicitGrabButton = kAnyButton;
   fButtonGrabView = nil;
   fPointerGrabType = kPGNoGrab;
   fGrabEventMask = 0;
   fOwnerEvents = true;
}

//______________________________________________________________________________
void EventTranslator::SetInputFocus(NSView<X11Window> *newFocusView)
{
   if (fFocusView && (fFocusView.fEventMask & kFocusChangeMask))
      Detail::SendFocusOutEvent(fEventQueue, fFocusView, kNotifyNormal);

   if (newFocusView) {
      if (newFocusView.fEventMask & kFocusChangeMask)
         Detail::SendFocusInEvent(fEventQueue, newFocusView, kNotifyNormal);

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

   if (fViewUnderPointer) {
      for (NSView<X11Window> *view  = fViewUnderPointer; view; view = view.fParentView) {
         if (view.fID == winID) {
            NSPoint location = {};
            location.x = fViewUnderPointer.fWidth / 2;
            location.y = fViewUnderPointer.fHeight / 2;
            location = [fViewUnderPointer convertPoint : location toView : nil];

            const Util::NSScopeGuard<FakeCrossingEvent> event([[FakeCrossingEvent alloc]
                                                               initWithWindow : [fViewUnderPointer window]
                                                               location : location]);
            if (!event.Get()) {
               //Hehe, if this happend, is it still possible to log????
               NSLog(@"EventTranslator::CheckUnmappedView, crossing event initialization failed");
               return;
            }

            Detail::SendLeaveEvent(fEventQueue, fViewUnderPointer, event.Get(), kNotifyNormal);
            fViewUnderPointer = nil;

            break;
         }
      }
   }

   ClearPointerIfViewIsRelated(fFocusView, winID);//TODO: send event to this view first?
   ClearPointerIfViewIsRelated(fKeyGrabView, winID);//TODO: send event to this view first??
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEventNoGrab(NSEvent *theEvent)
{
   //Without grab, things are simple: find a view which accepts pointer motion event.
   assert(theEvent != nil && "GeneratePointerMotionEventNoGrab, parameter 'theEvent' is nil");

   const Mask_t maskToTest = [NSEvent pressedMouseButtons] ?
                             (kPointerMotionMask | kButtonMotionMask) :
                             kPointerMotionMask;

   //Event without any emulated grab, receiver view can be "wrong" (result of Cocoa's "dragging").
   if (NSView<X11Window> *candidateView = FindViewForPointerEvent(theEvent)) {
      //Do propagation.
      candidateView = Detail::FindViewToPropagateEvent(candidateView, maskToTest);
      if (candidateView)//We have such a view, send event to a corresponding ROOT's window.
         Detail::SendPointerMotionEvent(fEventQueue, candidateView, theEvent);
   }
}

//______________________________________________________________________________
void EventTranslator::GeneratePointerMotionEventActiveGrab(NSEvent *theEvent)
{
   //More complex case. Grab can be result of button press and set by SetPointerGrab.
   //In case of button press (this is either passive->active or implicit grab),
   //Cocoa has it's own grab, so view (and window) can be not under cursor (but still
   //it receives events). So I can not simple use eventView here.

   //TODO: change interface? - remove eventView parameter declaration.

   if (!fButtonGrabView)//Implicit grab when nobody has PressButtonMask
      return;

   //assert(eventView != nil && "GeneratePointerMotionEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GeneratePointerMotionEventActiveGrab, parameter 'theEvent' is nil");

   const Mask_t maskToTest = [NSEvent pressedMouseButtons] ?
                             (kPointerMotionMask | kButtonMotionMask) :
                             kPointerMotionMask;

   if (fOwnerEvents) {
      //Complex case, we have to correctly report event.
      if (NSView<X11Window> *candidateView = FindViewForPointerEvent(theEvent)) {
         candidateView = Detail::FindViewToPropagateEvent(candidateView, maskToTest,
                                                          fButtonGrabView, fGrabEventMask);
         if (candidateView)//We have such a view, send event to a corresponding ROOT's window.
            Detail::SendPointerMotionEvent(fEventQueue, candidateView, theEvent);
      } else {
         //No such window - dispatch to the grab view.
         //Else: either implicit grab, or user requested grab with owner_grab == False.
         if (fGrabEventMask & maskToTest)
            Detail::SendPointerMotionEvent(fEventQueue, fButtonGrabView, theEvent);
      }
   } else {
      //Else: either implicit grab, or user requested grab with owner_grab == False.
      if (fGrabEventMask & maskToTest)
         Detail::SendPointerMotionEvent(fEventQueue, fButtonGrabView, theEvent);
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEventNoGrab(NSView<X11Window> *view, NSEvent *theEvent,
                                                     EMouseButton btn)
{
   //Generate button press event when no pointer grab is active:
   //either find a window with a passive grab, or create an implicit
   //grab (to emulate X11's behavior).

   assert(view != nil && "GenerateButtonPressEventNoGrab, parameter 'view' is nil");
   assert(theEvent != nil && "GenerateButtonPressEventNoGrab, parameter 'theEvent' is nil");

   FindButtonGrab(view, theEvent, btn);

   fImplicitGrabButton = btn;//This info is useless for any grab type except the implicit one.

   //Now we have to generate a sequence of enter/leave notify events,
   //like we "jump" from the previous view under the pointer to a grab view.

   Detail::GenerateCrossingEvents(fEventQueue, fViewUnderPointer, fButtonGrabView, theEvent, kNotifyGrab);

   //"Activate" a grab now, depending on type.
   if (fButtonGrabView) {
      if (fPointerGrabType == kPGPassiveGrab)
         [fButtonGrabView activatePassiveGrab];
      else if (fPointerGrabType == kPGImplicitGrab)
         [fButtonGrabView activateImplicitGrab];
   }

   //Send press event to a grab view (either passive grab or implicit,
   //but it has the required event bitmask).
   if (fButtonGrabView)
      Detail::SendButtonPressEvent(fEventQueue, fButtonGrabView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonPressEventActiveGrab(NSView<X11Window> * /*view*/, NSEvent *theEvent,
                                                         EMouseButton btn)
{
   //Generate button press event in the presence of activated pointer grab.

   //TODO: change interface? remove view parameter from declaration.

   //assert(view != nil && "GenerateButtonPressEventActiveGrab, view parameter is nil");
   assert(theEvent != nil && "GenerateButtonPressEventActiveGrab, parameter 'theEvent' is nil");

   //I did not find in X11 spec. the case when I have two passive grabs on window A and window B,
   //say left button on A and right button on B. What should happen if I press left button in A, move to
   //B and press the right button? In my test programm on X11 (Ubuntu) I can see, that now they BOTH
   //are active grabs. I'm not going to implement this mess, unless I have a correct formal description.
   if (!fButtonGrabView)
      return;

   if (fOwnerEvents) {
      if (NSView<X11Window> *candidateView = FindViewForPointerEvent(theEvent)) {
         //Do propagation.
         candidateView = Detail::FindViewToPropagateEvent(candidateView, kButtonPressMask,
                                                          fButtonGrabView, fGrabEventMask);
         //We have such a view, send an event to a corresponding ROOT's window.
         if (candidateView)
            Detail::SendButtonPressEvent(fEventQueue, candidateView, theEvent, btn);
      } else {
         if (fGrabEventMask & kButtonPressMask)
            Detail::SendButtonPressEvent(fEventQueue, fButtonGrabView, theEvent, btn);
      }
   } else {
      if (fGrabEventMask & kButtonPressMask)
         Detail::SendButtonPressEvent(fEventQueue, fButtonGrabView, theEvent, btn);
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent,
                                                       EMouseButton btn)
{
   //Generate button release event when there is no active pointer grab. Can this even happen??
   assert(eventView != nil && "GenerateButtonReleaseEventNoGrab, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEventNoGrabm parameter 'theEvent' is nil");

   if (NSView<X11Window> *candidateView = Detail::FindViewToPropagateEvent(eventView, kButtonReleaseMask))
      Detail::SendButtonReleaseEvent(fEventQueue, candidateView, theEvent, btn);
}

//______________________________________________________________________________
void EventTranslator::GenerateButtonReleaseEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent,
                                                           EMouseButton btn)
{
   //Generate button release event in the presence of active grab (explicit pointer grab, activated passive grab or implicit grab).

   assert(eventView != nil && "GenerateButtonReleaseEventActiveGrab, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateButtonReleaseEventActiveGrab, parameter 'theEvent' is nil");

   if (!fButtonGrabView) {
      //Still we have to cancel this grab (it's implicit grab on a root window).
      CancelPointerGrab();
      return;
   }

   //What if view is deleted in the middle of this function?
   const Util::NSStrongReference<NSView<X11Window> *> eventViewGuard(eventView);

   if (fButtonGrabView) {
      if (fOwnerEvents) {//X11: Either XGrabPointer with owner_events == True or passive grab (owner_events is always true)
         if (NSView<X11Window> *candidateView = FindViewForPointerEvent(theEvent)) {
            candidateView = Detail::FindViewToPropagateEvent(candidateView, kButtonReleaseMask,
                                                             fButtonGrabView, fGrabEventMask);
            //candidateView is either some view, or grab view, if its mask is ok.
            if (candidateView)
               Detail::SendButtonReleaseEvent(fEventQueue, candidateView, theEvent, btn);
         } else if (fGrabEventMask & kButtonReleaseMask)
            Detail::SendButtonReleaseEvent(fEventQueue, fButtonGrabView, theEvent, btn);
      } else {//Either implicit grab or GrabPointer with owner_events == False.
         if (fGrabEventMask & kButtonReleaseMask)
            Detail::SendButtonReleaseEvent(fEventQueue, fButtonGrabView, theEvent, btn);
      }
   } else {
      CancelPointerGrab();//root window had a grab, cancel it now.
   }

   if (fPointerGrabType == kPGPassiveGrab &&
       (btn == fButtonGrabView.fPassiveGrabButton || fButtonGrabView.fPassiveGrabButton == kAnyButton))
      CancelPointerGrab();

   if (fPointerGrabType == kPGImplicitGrab && btn == fImplicitGrabButton)
      CancelPointerGrab();
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyPressEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GenerateKeyPressEventNoGrab, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateKeyPressEventNoGrab, parameter 'theEvent' is nil");
   assert(fFocusView != nil && "GenerateKeyPressEventNoGrab, fFocusView is nil");

   FindKeyGrabView(eventView, theEvent);

   if (!fKeyGrabView) {
      NSView<X11Window> *candidateView = fFocusView;
      if (Detail::IsParent(fFocusView, eventView)) {
         //TODO: test theEvent.type? Can it be neither NSKeyDown nor NSKeyUp?
         NSView<X11Window> * const testView = Detail::FindViewToPropagateEvent(eventView, kKeyPressMask);

         if (testView && (testView == fFocusView || Detail::IsParent(fFocusView, testView)))
            candidateView = testView;
      }

      //TODO: test if focus (if it's chosen) want the event?
      GenerateKeyEventForView(candidateView, theEvent);
   } else
      GenerateKeyEventForView(fKeyGrabView, theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GenerateKeyEventActiveGrab, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateKeyEventActiveGrab, parameter 'theEvent' is nil");
   assert(fFocusView != nil && "GenerateKeyEventActiveGrab, fFocusView is nil");

   //TODO: assert on possible event types?
   const Mask_t eventMask = theEvent.type == NSKeyDown ? kKeyPressMask : kKeyReleaseMask;

   if (Detail::IsParent(fFocusView, eventView) || fFocusView == eventView) {
      NSView<X11Window> * const testView = Detail::FindViewToPropagateEvent(eventView, eventMask);
      if (testView && (testView == fFocusView || Detail::IsParent(fFocusView, testView)))
         GenerateKeyEventForView(testView, theEvent);
   } else
      GenerateKeyEventForView(fFocusView, theEvent);//Should I check the mask???

   if (theEvent.type == NSKeyUp && fKeyGrabView) {
      //Cancel grab?

      //NSString *characters = [theEvent charactersIgnoringModifiers];
      //assert(characters != nil && "GenerateKeyEventActiveGrab, [theEvent characters] returned nil");
      //assert([characters length] > 0 && "GenerateKeyEventActiveGrab, characters is an empty string");

      //Here I have a real trouble: on a key press GUI removes ... passive key grabs ...
      //this "does not affect any active grab", but later on a key release ... I'm not
      //able to find a grab to remove and can not ... cancel the grab.
      //I do it the same way it's done on Windows after all.
      //So, the condition was commented :(
      //if ([fKeyGrabView findPassiveKeyGrab : [characters characterAtIndex : 0]])
      fKeyGrabView = nil;//Cancel grab.
   }
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "GenerateKeyReleaseEventNoGrab, parameter 'eventView' is nil");
   assert(theEvent != nil && "GenerateKeyReleaseEventNoGrab, parameter 'theEvent' is nil");

   NSView<X11Window> *candidateView = fFocusView;

   if (eventView == fFocusView || Detail::IsParent(fFocusView, eventView)) {
      NSView<X11Window> * const testView = Detail::FindViewToPropagateEvent(eventView, kKeyReleaseMask);
      if (testView && (testView == fFocusView || Detail::IsParent(fFocusView, testView)))
         candidateView = testView;
   }

   //TODO: do I have to check if focus (if it was chosen) has a corresponding mask?
   GenerateKeyEventForView(candidateView, theEvent);
}

//______________________________________________________________________________
void EventTranslator::GenerateKeyEventForView(NSView<X11Window> *view, NSEvent *theEvent)
{
   //Generate key press event for a view without grab.
   assert(view != nil && "GenerateKeyEventForView, parameter 'view' is nil");
   assert(theEvent != nil && "GenerateKeyEventForView, parameter 'theEvent' is nil");
   assert(theEvent.type == NSKeyDown || theEvent.type == NSKeyUp &&
          "GenerateKeyEvenForView, event's type must be keydown or keyup");

   const Mask_t eventType = theEvent.type == NSKeyDown ? kKeyPressMask : kKeyReleaseMask;

   //TODO: this is not implemented, do I need it? (can require interface changes then).
   NSView<X11Window> *childView = nil;

   NSPoint mousePosition = {};
   if (QuartzWindow * const topLevel = FindWindowUnderPointer())
      mousePosition = [topLevel mouseLocationOutsideOfEventStream];

   if (eventType == kKeyPressMask)
      Detail::SendKeyPressEvent(fEventQueue, view, childView, theEvent, mousePosition);
   else
      Detail::SendKeyReleaseEvent(fEventQueue, view, childView, theEvent, mousePosition);
}

//______________________________________________________________________________
void EventTranslator::FindButtonGrab(NSView<X11Window> *fromView, NSEvent *theEvent, EMouseButton btn)
{
   //Find a view to become a grab view - either passive or implicit.

   assert(fromView != nil && "FindButtonGrabView, parameter 'fromView' is nil");
   assert(theEvent != nil && "FindButtonGrabView, parameter 'theEvent' is nil");

   assert(fPointerGrabType == kPGNoGrab && "FindButtonGrabView, grab is already activated");

   const unsigned keyModifiers = Detail::GetKeyboardModifiersFromCocoaEvent(theEvent);

   NSView<X11Window> *grabView = 0;
   NSView<X11Window> *buttonPressView = 0;

   for (NSView<X11Window> *view = fromView; view != nil; view = view.fParentView) {
      //Top-first view to receive button press event.
      if (!buttonPressView && (view.fEventMask & kButtonPressMask))
         buttonPressView = view;

      //Bottom-first view with passive grab.
      if (view.fPassiveGrabButton == kAnyButton || view.fPassiveGrabButton == btn) {
         //Check modifiers.
         if (view.fPassiveGrabKeyModifiers == kAnyModifier || (view.fPassiveGrabKeyModifiers & keyModifiers))
            grabView = view;
      }
   }

   if (grabView) {
      fButtonGrabView = grabView;
      fPointerGrabType = kPGPassiveGrab;
      fGrabEventMask = grabView.fPassiveGrabEventMask;
      fOwnerEvents = grabView.fPassiveGrabOwnerEvents;
   } else if (buttonPressView) {
      //This is an implicit grab.
      fButtonGrabView = buttonPressView;
      fPointerGrabType = kPGImplicitGrab;
      fGrabEventMask = buttonPressView.fEventMask;
      fOwnerEvents = false;
   } else {
      //Implicit grab with 'root' window?
      fButtonGrabView = nil;
      fPointerGrabType = kPGImplicitGrab;
      fGrabEventMask = 0;
      fOwnerEvents = false;
   }
}

//______________________________________________________________________________
void EventTranslator::FindKeyGrabView(NSView<X11Window> *eventView, NSEvent *theEvent)
{
   assert(eventView != nil && "FindKeyGrabView, parameter 'eventView' is nil");
   assert(theEvent != nil && "FindKeyGrabView, parameter 'theEvent' is nil");

   NSString * const characters = [theEvent charactersIgnoringModifiers];
   assert(characters != nil && "FindKeyGrabView, [theEvent characters] returned nil");
   assert([characters length] > 0 && "FindKeyGrabView, characters is an empty string");

   const unichar keyCode = [characters characterAtIndex : 0];
   const NSUInteger modifiers = [theEvent modifierFlags] & NSDeviceIndependentModifierFlagsMask;

   NSView<X11Window> *currentView = fFocusView;
   if (eventView != fFocusView && Detail::IsParent(fFocusView, eventView))
      currentView = eventView;

   for (; currentView; currentView = currentView.fParentView) {
      if ([currentView findPassiveKeyGrab : keyCode modifiers : modifiers])
         fKeyGrabView = currentView;
   }
}

}//X11
}//MacOSX
}//ROOT
