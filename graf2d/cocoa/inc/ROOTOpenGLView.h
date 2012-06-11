// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   26/04/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_ROOTOpenGLView
#define ROOT_ROOTOpenGLView

#import "X11Drawable.h"

@class QuartzView;

///////////////////////////////////////////
//                                       //
// OpenGL view's class.                  //
//                                       //
///////////////////////////////////////////

@interface ROOTOpenGLView : NSView <X11Window>

- (id) initWithFrame : (NSRect) frameRect pixelFormat : (NSOpenGLPixelFormat *) format;

//GL-view does not own GL-context, different GL contexts can be attached to the same view
//(though ROOT never does this). View has to know about GL-context only to notify it about
//geometry changes (calls -update method) and to clear drawable in a -dealloc method.

- (void) setOpenGLContext : (NSOpenGLContext *) context;

//ROOT's GL uses pixel format (TGLFormat class) when TGLWidget is
//created, after that, pixel format never changed (though I can do 
//this with ROOTOpenGLView, there is no interface in ROOT's GL code for this).
//So, pixel format is a property of ROOTOpenGLView. GL-view owns pixel format,
//it can also be reset externally (again, GL module never does this).
//Later, when creating GL-context, this pixel format is used (and
//ROOT creates GL-context per GL-widget, thus using pixel format from a widget.

- (NSOpenGLPixelFormat *) pixelFormat;
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat;

//View's geometry is updated by ROOT's GUI, but view
//can be hidden at the moment (for example, tab with GL-view is not active
//at the moment). If so, when view is visible again, context must
//be notified about changes in a drawable's geometry.
@property (nonatomic, assign) BOOL fUpdateContext;

//X11Drawable protocol
@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

- (int)      fX;
- (int)      fY;
- (unsigned) fWidth;
- (unsigned) fHeight;

//X11Window protocol.

- (void) getAttributes : (WindowAttributes_t *) attr;

- (void) setDrawableSize : (NSSize) newSize;
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void) setX : (int) x Y : (int) y;

@property (nonatomic, assign) long  fEventMask;
@property (nonatomic, assign) int   fClass;
@property (nonatomic, readonly) int fMapState;
@property (nonatomic, assign) int   fDepth;
@property (nonatomic, assign) int   fBitGravity;
@property (nonatomic, assign) int   fWinGravity;

@property (nonatomic, assign) QuartzPixmap *fBackBuffer;//nil.

@property (nonatomic, assign) QuartzView          *fParentView;
@property (nonatomic, assign) unsigned             fLevel;
@property (nonatomic, readonly) NSView<X11Window> *fContentView;
@property (nonatomic, readonly) QuartzWindow      *fQuartzWindow;

@property (nonatomic, assign) int      fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL     fOwnerEvents;

- (void) mapWindow;
- (void) mapSubwindows;
- (void) configureNotifyTree;
- (BOOL) fIsOverlapped;
- (void) setOverlapped : (BOOL) overlap;
- (void) updateLevel : (unsigned) newLevel;

/*
- (void) setAttributes : (const SetWindowAttributes_t *) attr;
- (void) mapRaised;
- (void) mapSubwindows;
- (void) unmapWindow;
- (void) raiseWindow;
- (void) lowerWindow;
*/

- (void) addPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (void) removePassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode;

//Cursors.
@property (nonatomic, assign) ECursor fCurrentCursor;

@end

namespace ROOT {
namespace MacOSX {
namespace OpenGL {

bool GLViewIsValidDrawable(ROOTOpenGLView *glView);

}
}
}

#endif
