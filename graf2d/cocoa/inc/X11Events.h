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

@class QuartzWindow;
@class QuartzView;
@class NSEvent;

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

   void GenerateConfigureNotifyEvent(QuartzView *view, const NSRect &newFrame);
   void GenerateDestroyNotify(unsigned winID);
   void GenerateExposeEvent(QuartzView *view, const NSRect &exposedRect);

   void GenerateCrossingEvent(QuartzView *viewUnderPointer, NSEvent *theEvent);
   void GeneratePointerMotionEvent(QuartzView *eventView, NSEvent *theEvent);
   void GenerateButtonPressEvent(QuartzView *eventView, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonReleaseEvent(QuartzView *eventView, NSEvent *theEvent, EMouseButton btn);
   
   void GenerateKeyPressEvent(QuartzView *eventView, NSEvent *theEvent);
   
   void SetPointerGrab(QuartzView *grabView, unsigned eventMask, bool ownerEvents);
   void CancelPointerGrab();
   
   //Window winID was either deleted or unmapped.
   //If it's a grab view or a parent of a grab view - cancel grab.
   //If it's a "view under pointer" - reset view under pointer.
   void CheckUnmappedView(Window_t winID);

private:
   bool HasPointerGrab()const;


   //Used both by grab and non-grab case.
   void GenerateCrossingEvent(QuartzView *viewUnderPointer, NSEvent *theEvent, EXMagic detail);
   void GenerateCrossingEventActiveGrab(QuartzView *eventView, NSEvent *theEvent);

   void GeneratePointerMotionEventNoGrab(QuartzView *view, NSEvent *theEvent);
   void GeneratePointerMotionEventActiveGrab(QuartzView *eventView, NSEvent *theEvent);

   void GenerateButtonPressEventNoGrab(QuartzView *view, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonPressEventActiveGrab(QuartzView *view, NSEvent *theEvent, EMouseButton btn);

   void GenerateButtonReleaseEventNoGrab(QuartzView *eventView, NSEvent *theEvent, EMouseButton btn);
   void GenerateButtonReleaseEventActiveGrab(QuartzView *eventView, NSEvent *theEvent, EMouseButton btn);
   
   void GenerateKeyPressEventNoGrab(QuartzView *view, NSEvent *theEvent);

   void FindGrabView(QuartzView *fromView, NSEvent *theEvent, EMouseButton btn);
   Ancestry FindRelation(QuartzView *view1, QuartzView *view2, QuartzView **lca);
   void SortTopLevelWindows();
   QuartzWindow *FindTopLevelWindowForMouseEvent();

   QuartzView *fViewUnderPointer;
   std::vector<QuartzView *> fBranch1;
   std::vector<QuartzView *> fBranch2;
   
   PointerGrab fPointerGrab;
   unsigned fGrabEventMask;
   bool fOwnerEvents;


   QuartzView *fCurrentGrabView;
   
   std::vector<QuartzWindow *> fWindowStack;
};

void MapUnicharToKeySym(unichar key, char *buf, Int_t len, UInt_t &rootKeySym);
Int_t MapKeySymToKeyCode(Int_t keySym);

}//X11
}//MacOSX
}//ROOT

#endif
