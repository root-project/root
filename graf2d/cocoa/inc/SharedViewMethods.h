// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   26/04/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_SharedViewMethods
#define ROOT_SharedViewMethods

/////////////////////////////////////////////////////////////////////////////////
//                                                                             //
// Inclusion guards are no really needed, this is unusuall header file.        //
// Since I can not inherit ROOTOpenGLView from QuartzView and have to          //
// inherit NSOpenGLView, QuartzView and ROOTOpenGLView are two completely      //
// unrelated classes (and can not have a common custom view class as base).    //
// But still they have a lot in common (event processing, etc.),               //
// this common part - "shared methods".                                        //
//                                                                             //
/////////////////////////////////////////////////////////////////////////////////


//X11Window protocol.

//______________________________________________________________________________
- (int) fX
{
   return self.frame.origin.x;
}

//______________________________________________________________________________
- (int) fY
{
   return self.frame.origin.y;
}

//______________________________________________________________________________
- (unsigned) fWidth
{
   return self.frame.size.width;
}

//______________________________________________________________________________
- (unsigned) fHeight
{
   return self.frame.size.height;
}


//TODO: check, if it can be used by OpenGLView.
//______________________________________________________________________________
- (void) setDrawableSize : (NSSize) newSize
{
   assert(!(newSize.width < 0) && "setDrawableSize, width is negative");
   assert(!(newSize.height < 0) && "setDrawableSize, height is negative");
   
   //This will cause redraw(?)
   
   //In X11, resize changes the size, but upper-left corner is not changed.
   //In Cocoa, bottom-left is fixed.
   NSRect frame = self.frame;
   frame.size = newSize;
   
   self.frame = frame;
}

//______________________________________________________________________________
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h
{
   assert(fParentView != nil && "setX:Y:width:height:, parent view is nil");

   NSRect newFrame = {};
   newFrame.origin.x = x;
   newFrame.origin.y = y;
   newFrame.size.width = w;
   newFrame.size.height = h;
   
   self.frame = newFrame;
}

//______________________________________________________________________________
- (void) setX : (int) x Y : (int) y
{
   assert(fParentView != nil && "setX:Y:, parent view is nil");
   
   NSRect newFrame = self.frame;
   newFrame.origin.x = x;
   newFrame.origin.y = y;
   
   self.frame = newFrame;
}

//______________________________________________________________________________
- (int) fMapState
{
   if ([self isHidden])
      return kIsUnmapped;

   for (QuartzView *parent = fParentView; parent; parent = parent.fParentView) {
      if ([parent isHidden])
         return kIsUnviewable;
   }

   return kIsViewable;
}


//______________________________________________________________________________
- (NSView<X11Window> *) fContentView
{
   return self;
}

//______________________________________________________________________________
- (QuartzWindow *) fQuartzWindow
{
   return (QuartzWindow *)[self window];
}

//Events.

//______________________________________________________________________________
- (void) mouseDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseDown, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseDown, gVirtualX is either null or has a type, different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton1);
}

//______________________________________________________________________________
- (void) scrollWheel : (NSEvent*) theEvent
{
   assert(fID != 0 && "scrollWheel, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "scrollWheel, gVirtualX is either null or has a type, different from TGCocoa");

   const CGFloat deltaY = [theEvent deltaY];
   if (deltaY < 0) {
      vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton5);
      vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton5);
   } else if (deltaY > 0) {
      vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton4);
      vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton4);
   }
}

//______________________________________________________________________________
- (void) rightMouseDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "rightMouseDown, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "rightMouseDown, gVirtualX is either null or has type different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton3);
}

//______________________________________________________________________________
- (void) mouseUp : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseUp, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx && "mouseUp, gVirtualX is either null or has type different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton1);
}

//______________________________________________________________________________
- (void) rightMouseUp : (NSEvent *) theEvent
{

   assert(fID != 0 && "rightMouseUp, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "rightMouseUp, gVirtualX is either null or has type different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton2);
}

//______________________________________________________________________________
- (void) mouseEntered : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseEntered, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseEntered, gVirtualX is null or not of TGCocoa type");

   vx->GetEventTranslator()->GenerateCrossingEvent(self, theEvent);  
}

//______________________________________________________________________________
- (void) mouseExited : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseExited, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseExited, gVirtualX is null or not of TGCocoa type");

   vx->GetEventTranslator()->GenerateCrossingEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) mouseMoved : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseMoved, fID is 0");
   
   if (fParentView)//Suppress events in all views, except the top-level one.
      return;

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) mouseDragged : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseDragged, fID is 0");
   
   //mouseMoved and mouseDragged work differently 
   //(drag events are generated only for one view, where drag started).
   //if (fParentView)
   //   return;
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);   
}

//______________________________________________________________________________
- (void) rightMouseDragged : (NSEvent *) theEvent
{
   assert(fID != 0 && "rightMouseDragged, fID is 0");
   
   //mouseMoved and mouseDragged work differently 
   //(drag events are generated only for one view, where drag started).
   //if (fParentView)
   //   return;
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "rightMouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);   
}

//______________________________________________________________________________
- (void) keyDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "keyDown, fID is 0");
  
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "keyDown, gVirtualX is null or not of TGCocoa type");
   vx->GetEventTranslator()->GenerateKeyPressEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) keyUp:(NSEvent *)theEvent
{
   (void)theEvent;
}

//First responder staff.

//______________________________________________________________________________
- (BOOL) acceptsFirstMouse : (NSEvent *)theEvent
{
   (void)theEvent;
   return YES;
}

//______________________________________________________________________________
- (BOOL) acceptsFirstResponder
{
   return YES;
}

//______________________________________________________________________________
- (BOOL) becomeFirstResponder
{
   //Change focus.
   NSView<X11Window> *focusView = nil;
   for (NSView<X11Window> *view = self; view; view = view.fParentView) {
      if (view.fEventMask & kFocusChangeMask) {
         focusView = view;
         break;
      }
   }

   if (!focusView)
      focusView = ((QuartzWindow *)[self window]).fContentView;
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "becomeFirstResponder, gVirtualX is null or not of TGCocoa type");
   vx->GetEventTranslator()->GenerateFocusChangeEvent(focusView);

   return YES;
}

//______________________________________________________________________________
- (BOOL) resignFirstResponder
{
   //Change focus.
   //NSResponder returns YES, so do I.
   return YES;
}

//______________________________________________________________________________
- (void) addPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers
{
   //Remove and add (not to traverse twice).
   [self removePassiveKeyGrab : keyCode modifiers : modifiers];
   PassiveKeyGrab *newGrab = [[PassiveKeyGrab alloc] initWithKey : keyCode modifiers : modifiers];
   [fPassiveKeyGrabs addObject : newGrab];
   [newGrab release];
}

//______________________________________________________________________________
- (void) removePassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers
{
   const NSUInteger count = [fPassiveKeyGrabs count];
   for (NSUInteger i = 0; i < count; ++i) {
      PassiveKeyGrab *grab = [fPassiveKeyGrabs objectAtIndex : i];
      if ([grab matchKey : keyCode modifiers : modifiers]) {
         [fPassiveKeyGrabs removeObjectAtIndex : i];
         break;
      }
   }
}

//______________________________________________________________________________
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers
{
   NSEnumerator *enumerator = [fPassiveKeyGrabs objectEnumerator];
   while (PassiveKeyGrab *grab = (PassiveKeyGrab *)[enumerator nextObject]) {
      if ([grab matchKey : keyCode modifiers : modifiers])
         return grab;
   }

   return nil;
}

//______________________________________________________________________________
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode
{
   //Do not check modifiers.
   NSEnumerator *enumerator = [fPassiveKeyGrabs objectEnumerator];
   while (PassiveKeyGrab *grab = (PassiveKeyGrab *)[enumerator nextObject]) {
      if ([grab matchKey : keyCode])
         return grab;
   }

   return nil;
}

#endif
