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

@interface ROOTOpenGLView : NSOpenGLView <X11Window>

//NSOpenGLView methods.
- (id) initWithFrame : (NSRect) frameRect pixelFormat : (NSOpenGLPixelFormat *) format;
- (void) clearGLContext;
- (NSOpenGLContext *) openGLContext;
- (NSOpenGLPixelFormat *) pixelFormat;
- (void) prepareOpenGL;
- (void) reshape;
- (void) setOpenGLContext : (NSOpenGLContext *) context;
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat;
- (void) update;

//X11Drawable protocol

@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

- (int)      fX;
- (int)      fY;
- (unsigned) fWidth;
- (unsigned) fHeight;

//X11Window protocol.

- (void) setDrawableSize : (NSSize) newSize;
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void) setX : (int) x Y : (int) y;

@property (nonatomic, assign) long  fEventMask;
@property (nonatomic, readonly) int fMapState;

@property (nonatomic, assign) QuartzView          *fParentView;
@property (nonatomic, assign) unsigned             fLevel;
@property (nonatomic, readonly) NSView<X11Window> *fContentView;
@property (nonatomic, readonly) QuartzWindow      *fQuartzWindow;

@property (nonatomic, assign) int      fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL     fOwnerEvents;
/*
- (void) getAttributes : (WindowAttributes_t *) attr;
- (void) setAttributes : (const SetWindowAttributes_t *) attr;

- (void) mapRaised;
- (void) mapWindow;
- (void) mapSubwindows;
- (void) unmapWindow;
- (void) raiseWindow;
- (void) lowerWindow;

- (BOOL) fIsOverlapped;
- (void) setOverlapped : (BOOL) overlap;
- (void) updateLevel : (unsigned) newLevel;
- (void) configureNotifyTree;
*/
- (void) addPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (void) removePassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode;

//Cursors.
@property (nonatomic, assign) ECursor fCurrentCursor;

@end

#endif
