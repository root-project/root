// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_X11Events
#define ROOT_X11Events

#include <deque>

#include "GuiTypes.h"

#include <Foundation/Foundation.h>

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// EventTranslator class translates Cocoa events to 'ROOT's X11' events.//
// In 90% cases there is no direct mapping from Cocoa event to          //
// X11 event: Cocoa events are more simple (from programmer's POV).     //
// EventTranslator tries to emulate X11 behavior.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

class TGCocoa;

@protocol X11Window;

@class QuartzWindow;
@class NSEvent;
@class NSView;

namespace ROOT {
namespace MacOSX {
namespace X11 {//X11 emulation for Cocoa.

enum PointerGrab {
   kPGNoGrab,
   kPGImplicitGrab,
   kPGActiveGrab,
   kPGPassiveGrab
};

typedef std::deque<Event_t> EventQueue_t;

class EventTranslator {
   friend class ::TGCocoa;
public:
   EventTranslator();

   void GenerateConfigureNotifyEvent(NSView<X11Window> *view, const NSRect &newFrame);
   void GenerateDestroyNotify(unsigned /*winID*/);
   void GenerateExposeEvent(NSView<X11Window> *view, const NSRect &exposedRect);

   void GenerateCrossingEvent(NSEvent *theEvent);
   void GeneratePointerMotionEvent(NSEvent *theEvent);

   void GenerateButtonPressEvent(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonReleaseEvent(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);

   void GenerateKeyPressEvent(NSView<X11Window> *eventView, NSEvent *theEvent);
   void GenerateKeyReleaseEvent(NSView<X11Window> *eventView, NSEvent *theEvent);

   void GenerateFocusChangeEvent(NSView<X11Window> *eventView);

   void SetPointerGrab(NSView<X11Window> *grabView, unsigned eventMask, bool ownerEvents);
   void CancelPointerGrab();

   void SetInputFocus(NSView<X11Window> *focusView);
   unsigned GetInputFocus()const;

   //Window winID was either deleted or unmapped.
   //If it's a grab view or a parent of a grab view - cancel grab.
   //If it's a "view under pointer" - reset view under pointer.
   //If it's a focus view, cancel focus.
   void CheckUnmappedView(Window_t winID);

   bool HasPointerGrab()const;

private:

   //Used both by grab and non-grab case.
   void GenerateCrossingEventNoGrab(NSEvent *theEvent);
   void GenerateCrossingEventActiveGrab(NSEvent *theEvent);

   void GeneratePointerMotionEventNoGrab(NSEvent *theEvent);
   void GeneratePointerMotionEventActiveGrab(NSEvent *theEvent);

   void GenerateButtonPressEventNoGrab(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonPressEventActiveGrab(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn);

   void GenerateButtonReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonReleaseEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);

   void GenerateKeyPressEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent);
   void GenerateKeyReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent);

   void GenerateKeyEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent);//Both press/release events.
   void GenerateKeyEventForView(NSView<X11Window> *view, NSEvent *theEvent);//Both press/release events.

   void FindButtonGrab(NSView<X11Window> *fromView, NSEvent *theEvent, EMouseButton btn);
   void FindKeyGrabView(NSView<X11Window> *eventView, NSEvent *theEvent);

   NSView<X11Window> *fViewUnderPointer;

   PointerGrab fPointerGrabType;
   unsigned fGrabEventMask;
   bool fOwnerEvents;


   NSView<X11Window> *fButtonGrabView;
   NSView<X11Window> *fKeyGrabView;
   NSView<X11Window> *fFocusView;
   EMouseButton fImplicitGrabButton;

   EventQueue_t fEventQueue;
};

void MapUnicharToKeySym(unichar key, char *buf, Int_t len, UInt_t &rootKeySym);
Int_t MapKeySymToKeyCode(Int_t keySym);
NSUInteger GetCocoaKeyModifiersFromROOTKeyModifiers(UInt_t rootKeyModifiers);

UInt_t GetModifiers();//Mouse buttons + keyboard modifiers.

}//X11
}//MacOSX
}//ROOT

#endif
