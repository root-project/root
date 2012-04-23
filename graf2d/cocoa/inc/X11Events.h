//Author: Timur Pocheptsov 16/02/2012

#ifndef ROOT_X11Events
#define ROOT_X11Events

#include <vector>

#ifndef ROOT_GuiTypes
#include "GuiTypes.h"
#endif

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// EventTranslator class translates Cocoa events to 'ROOT's X11' events.//
// In 90% cases there is no direct mapping from Cocoa event to          //
// X11 event: Cocoa events are more simple (from programmer's POV).     //
// EventTranslator tries to emulate X11 behavior.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

@protocol X11Window;

@class QuartzWindow;
@class NSEvent;
@class NSView;

namespace ROOT {
namespace MacOSX {
namespace X11 {//X11 emulation for Cocoa.

enum class Ancestry {
   view1IsParent,
   view2IsParent,
   haveNonRootAncestor,
   ancestorIsRoot
};

enum class PointerGrab {
   noGrab,
   implicitGrab,
   activeGrab,
   passiveGrab
};

class EventTranslator {

public:
   EventTranslator();

   void GenerateConfigureNotifyEvent(NSView<X11Window> *view, const NSRect &newFrame);
   void GenerateDestroyNotify(unsigned winID);
   void GenerateExposeEvent(NSView<X11Window> *view, const NSRect &exposedRect);

   void GenerateCrossingEvent(NSView<X11Window> *viewUnderPointer, NSEvent *theEvent);
   void GeneratePointerMotionEvent(NSView<X11Window> *eventView, NSEvent *theEvent);
   
   //TODO: instead of passing EMouseButton, use info from NSEvent???
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

private:
   bool HasPointerGrab()const;

   //Used both by grab and non-grab case.
   void GenerateCrossingEvent(NSView<X11Window> *viewUnderPointer, NSEvent *theEvent, EXMagic detail);
   void GenerateCrossingEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent);

   void GeneratePointerMotionEventNoGrab(NSView<X11Window> *view, NSEvent *theEvent);
   void GeneratePointerMotionEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent);

   void GenerateButtonPressEventNoGrab(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonPressEventActiveGrab(NSView<X11Window> *view, NSEvent *theEvent, EMouseButton btn);

   void GenerateButtonReleaseEventNoGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonReleaseEventActiveGrab(NSView<X11Window> *eventView, NSEvent *theEvent, EMouseButton btn);
   
   void GenerateKeyPressEventNoGrab(NSEvent *theEvent);
   void GenerateKeyReleaseEventNoGrab(NSEvent *theEvent);
   
   void GenerateKeyEventActiveGrab(NSEvent *theEvent);//Both press/release events.
   void GenerateKeyEventForView(NSView<X11Window> *view, NSEvent *theEvent);//Both press/release events.

   void FindButtonGrabView(NSView<X11Window> *fromView, NSEvent *theEvent, EMouseButton btn);
   void FindKeyGrabView(NSView<X11Window> *fromView, NSEvent *theEvent);
   NSView<X11Window> *FindViewUnderPointer();
   
   Ancestry FindRelation(NSView<X11Window> *view1, NSView<X11Window> *view2, NSView<X11Window> **lca);
   void SortTopLevelWindows();
   QuartzWindow *FindTopLevelWindowForMouseEvent();

   NSView<X11Window> *fViewUnderPointer;
   std::vector<NSView<X11Window> *> fBranch1;
   std::vector<NSView<X11Window> *> fBranch2;
   
   PointerGrab fPointerGrab;
   unsigned fGrabEventMask;
   bool fOwnerEvents;


   NSView<X11Window> *fButtonGrabView;
   NSView<X11Window> *fKeyGrabView;
   NSView<X11Window> *fFocusView;
   
   std::vector<QuartzWindow *> fWindowStack;
};

void MapUnicharToKeySym(unichar key, char *buf, Int_t len, UInt_t &rootKeySym);
Int_t MapKeySymToKeyCode(Int_t keySym);
NSUInteger GetCocoaKeyModifiersFromROOTKeyModifiers(UInt_t rootKeyModifiers);

}//X11
}//MacOSX
}//ROOT

#endif
