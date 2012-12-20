// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_X11Drawable
#define ROOT_X11Drawable

#import <Cocoa/Cocoa.h>

#import "CocoaGuiTypes.h"
#import "TVirtualX.h"
#import "GuiTypes.h"

@class PassiveKeyGrab;
@class QuartzWindow;
@class QuartzPixmap;
@class QuartzImage;
@class QuartzView;

/////////////////////////////////////////////////////////////////////////////////////
//                                                                                 //
// Protocol for "drawables". It can be window, view (child window), pixmap.        //
// X11Drawable is a generic part for both windows and pixmaps.                     //
//                                                                                 //
/////////////////////////////////////////////////////////////////////////////////////

@protocol X11Drawable 
@optional

@property (nonatomic, assign) unsigned fID;   //Drawable's id for GUI and TGCocoa.

//In X11 drawable is a window or a pixmap, ROOT's GUI
//also has this ambiguity. So I have a property
//to check in TGCocoa, what's the object.
- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

//Either [[NSGraphicsContext currentContext] graphicsPort]
//or bitmap context (pixmap).
@property (nonatomic, readonly) CGContextRef  fContext;

//Readonly geometry:
- (int)      fX;
- (int)      fY; //top-left corner system.
- (unsigned) fWidth;
- (unsigned) fHeight;

//Functions to copy one drawable into another.
- (void) copy : (NSObject<X11Drawable> *) src area : (ROOT::MacOSX::X11::Rectangle) area withMask : (QuartzImage *)mask
         clipOrigin : (ROOT::MacOSX::X11::Point) origin toPoint : (ROOT::MacOSX::X11::Point) dstPoint;

//Get access to pixel data.
- (unsigned char *) readColorBits : (ROOT::MacOSX::X11::Rectangle) area;

@end

@protocol X11Window <X11Drawable>
@optional

//Geometry setters:
- (void) setDrawableSize : (NSSize) newSize;
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void) setX : (int) x Y : (int) y;

//I have to somehow emulate X11's behavior to make ROOT's GUI happy,
//that's why I have this bunch of properties here to be set/read from a window.
//Some of them are used, some are just pure "emulation".
//Properties, which are used, are commented in a declaration.

/////////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t

@property (nonatomic, assign) long          fEventMask; //Specifies which events must be processed by widget.
@property (nonatomic, assign) int           fClass;
@property (nonatomic, assign) int           fDepth;
@property (nonatomic, assign) int           fBitGravity;
@property (nonatomic, assign) int           fWinGravity;
@property (nonatomic, assign) unsigned long fBackgroundPixel;//Used by TGCocoa::ClearArea.
@property (nonatomic, retain) QuartzImage  *fBackgroundPixmap;//Hmm, image, pixmap ...
@property (nonatomic, readonly) int         fMapState;
@property (nonatomic, assign) BOOL          fOverrideRedirect;

//End of SetWindowAttributes_t/WindowAttributes_t
/////////////////////////////////////////////////////////////////


//"Back buffer" is a bitmap, used by canvas window (only).
@property (nonatomic, retain) QuartzPixmap *fBackBuffer;
//Parent view can be only QuartzView.
@property (nonatomic, assign) QuartzView *fParentView;
//Window has a content view, self is a content view for a view.
//I NSView is a parent for QuartzView and ROOTOpenGLView.
@property (nonatomic, readonly) NSView<X11Window> *fContentView;
@property (nonatomic, readonly) QuartzWindow      *fQuartzWindow;

//Passive button grab emulation.
//ROOT's GUI does not use several passive button
//grabs on the same window, so no containers,
//just one grab.
@property (nonatomic, assign) int      fPassiveGrabButton;
@property (nonatomic, assign) unsigned fPassiveGrabEventMask;
@property (nonatomic, assign) unsigned fPassiveGrabKeyModifiers;

@property (nonatomic, assign) unsigned fActiveGrabEventMask;

@property (nonatomic, assign) BOOL     fPassiveGrabOwnerEvents;

- (void) activatePassiveGrab;
- (void) activateImplicitGrab;
- (void) activateGrab : (unsigned) eventMask ownerEvents : (BOOL) ownerEvents;
- (void) cancelGrab;

- (BOOL) acceptsCrossingEvents : (unsigned) eventMask;

//Nested views ("windows").
//Child can be any view, inherited
//from NSView adopting X11Window protocol.
- (void) addChild : (NSView<X11Window> *) child;

//X11/ROOT GUI's attributes
- (void) getAttributes : (WindowAttributes_t *) attr;
- (void) setAttributes : (const SetWindowAttributes_t *) attr;

//X11's XMapWindow etc.
- (void) mapRaised;
- (void) mapWindow;
- (void) mapSubwindows;
- (void) unmapWindow;
- (void) raiseWindow;
- (void) lowerWindow;

- (BOOL) fIsOverlapped;
- (void) setOverlapped : (BOOL) overlap;
- (void) configureNotifyTree;

- (void) addPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (void) removePassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode;

//Cursors.
@property (nonatomic, assign) ECursor fCurrentCursor;

@property (nonatomic, assign) BOOL fIsDNDAware;

//"Properties" (X11 properties)
- (void) setProperty : (const char *) propName data : (unsigned char *) propData size : (unsigned) dataSize 
         forType : (Atom_t) dataType format : (unsigned) format;
- (BOOL) hasProperty : (const char *) propName;
- (unsigned char *) getProperty : (const char *) propName returnType : (Atom_t *) type 
   returnFormat : (unsigned *) format nElements : (unsigned *) nElements;
- (void) removeProperty : (const char *) propName;

@end

#endif
