// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_QuartzWindow
#define ROOT_QuartzWindow

#import <Cocoa/Cocoa.h>

#import "X11Drawable.h"
#import "GuiTypes.h"

////////////////////////////////////////////////
//                                            //
// QuartzWindow class : top-level window.     //
//                                            //
////////////////////////////////////////////////

@interface QuartzWindow : NSWindow<X11Window, NSWindowDelegate>

//Life-cycle: "ctor".
- (id) initWithContentRect : (NSRect) contentRect styleMask : (NSUInteger) windowStyle 
       backing : (NSBackingStoreType) bufferingType defer : (BOOL) deferCreation
       windowAttributes : (const SetWindowAttributes_t *) attr;

//This is to emulate "transient" window/main window relationship:
@property (nonatomic, assign) QuartzWindow *fMainWindow;
- (void) addTransientWindow : (QuartzWindow *) window;
//1. X11Drawable protocol.

@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

@property (nonatomic, readonly) CGContextRef  fContext;

//Geometry.
- (int) fX;
- (int) fY;

- (unsigned) fWidth;
- (unsigned) fHeight;

- (void) setDrawableSize : (NSSize) newSize;
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void) setX : (int) x Y : (int) y;

//
- (void) copy : (NSObject<X11Drawable> *) src area : (Rectangle_t) area withMask : (QuartzImage *) mask 
         clipOrigin : (Point_t) origin toPoint : (Point_t) dstPoint;

- (unsigned char *) readColorBits : (Rectangle_t) area;


//X11Window protocol.

/////////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t

@property (nonatomic, assign) long          fEventMask;
@property (nonatomic, assign) int           fClass;
@property (nonatomic, assign) int           fDepth;
@property (nonatomic, assign) int           fBitGravity;
@property (nonatomic, assign) int           fWinGravity;
@property (nonatomic, assign) unsigned long fBackgroundPixel;
@property (nonatomic, readonly) int         fMapState;

//End of SetWindowAttributes_t/WindowAttributes_t
/////////////////////////////////////////////////////////////////

//"Back buffer" is a bitmap, attached to a window by TCanvas.
@property (nonatomic, assign) QuartzPixmap          *fBackBuffer;
@property (nonatomic, assign) QuartzView            *fParentView;
@property (nonatomic, readonly) NSView<X11Window>   *fContentView;
@property (nonatomic, readonly) QuartzWindow        *fQuartzWindow;

@property (nonatomic, assign) int      fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL     fOwnerEvents;

//Children subviews.
- (void) addChild : (NSView<X11Window> *) child;

//X11/ROOT GUI's attributes.
- (void) getAttributes : (WindowAttributes_t *) attr;
- (void) setAttributes : (const SetWindowAttributes_t *) attr;

//X11's XMapWindow etc.
- (void) mapRaised;
- (void) mapWindow;
- (void) mapSubwindows;
- (void) unmapWindow;

//Cursors.
@property (nonatomic, assign) ECursor fCurrentCursor;

@end

//////////////////////////////////////////////////////////////
//                                                          //
// I have to attach passive key grabs to a view.            //
//                                                          //
//////////////////////////////////////////////////////////////

@interface PassiveKeyGrab : NSObject
- (unichar)    fKeyCode;
- (NSUInteger) fModifiers;
- (id) initWithKey : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (BOOL) matchKey : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (BOOL) matchKey : (unichar) keyCode;
@end

////////////////////////////////////////
//                                    //
// QuartzView class - child window.   //
//                                    //
////////////////////////////////////////

@class QuartzImage;

@interface QuartzView : NSView<X11Window>

//Life-cycle.
- (id) initWithFrame : (NSRect) frame windowAttributes : (const SetWindowAttributes_t *) attr;

//Clip mask - to deal with overlaps.
@property (nonatomic, assign) BOOL fClipMaskIsValid;
- (BOOL) initClipMask;
- (QuartzImage *) fClipMask;
- (void) addOverlap : (NSRect)overlapRect;

//X11Drawable protocol.

@property (nonatomic, assign) unsigned fID;

- (BOOL) fIsPixmap;
- (BOOL) fIsOpenGLWidget;

@property (nonatomic, assign) CGContextRef fContext;

//Geometry.
- (int)      fX;
- (int)      fY;
- (unsigned) fWidth;
- (unsigned) fHeight;
- (void)     setDrawableSize : (NSSize) newSize;
- (void)     setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void)     setX : (int) x Y : (int) y;

- (void)     copy : (NSObject<X11Drawable> *) src area : (Rectangle_t) area withMask : (QuartzImage *)mask 
             clipOrigin : (Point_t) origin toPoint : (Point_t) dstPoint;
- (unsigned char *) readColorBits : (Rectangle_t) area;

//X11Window protocol.

/////////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t

@property (nonatomic, assign) long          fEventMask;
@property (nonatomic, assign) int           fClass;
@property (nonatomic, assign) int           fDepth;
@property (nonatomic, assign) int           fBitGravity;
@property (nonatomic, assign) int           fWinGravity;
@property (nonatomic, assign) unsigned long fBackgroundPixel;
@property (nonatomic, readonly) int         fMapState;

//End of SetWindowAttributes_t/WindowAttributes_t
/////////////////////////////////////////////////////////////////


@property (nonatomic, assign) QuartzPixmap        *fBackBuffer;
@property (nonatomic, assign) QuartzView          *fParentView;
@property (nonatomic, assign) unsigned             fLevel;
@property (nonatomic, readonly) NSView<X11Window> *fContentView;
@property (nonatomic, readonly) QuartzWindow      *fQuartzWindow;

@property (nonatomic, assign) int      fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL     fOwnerEvents;

//Children subviews.
- (void) addChild : (NSView<X11Window> *)child;

//X11/ROOT GUI's attributes.
- (void) getAttributes : (WindowAttributes_t *) attr;
- (void) setAttributes : (const SetWindowAttributes_t *) attr;

//
- (void) mapRaised;
- (void) mapWindow;
- (void) mapSubwindows;

- (void) unmapWindow;
//
- (void) raiseWindow;
- (void) lowerWindow;
//

- (BOOL) fIsOverlapped;
- (void) setOverlapped : (BOOL) overlap;
- (void) updateLevel : (unsigned) newLevel;
- (void) configureNotifyTree;

//Additional methods and properties.

@property (nonatomic, assign) BOOL   fSnapshotDraw;
- (BOOL) isFlipped;//override method from NSView.

//Keyboard:
- (void) addPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (void) removePassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode modifiers : (NSUInteger) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (unichar) keyCode;

//Cursors.
@property (nonatomic, assign) ECursor fCurrentCursor;

@end


//Aux. functions.
namespace ROOT {
namespace MacOSX {
namespace X11 {

QuartzWindow *CreateTopLevelWindow(Int_t x, Int_t y, UInt_t w, UInt_t h, UInt_t border, Int_t depth,
                                   UInt_t clss, void *visual, SetWindowAttributes_t *attr, UInt_t);
QuartzView *CreateChildView(QuartzView *parent, Int_t x, Int_t y, UInt_t w, UInt_t h, UInt_t border, Int_t depth,
                            UInt_t clss, void *visual, SetWindowAttributes_t *attr, UInt_t wtype);

void GetRootWindowAttributes(WindowAttributes_t *attr);
void GetWindowAttributes(NSObject<X11Window> *window, WindowAttributes_t *dst);

//Coordinate conversion.
int GlobalYCocoaToROOT(CGFloat yCocoa);
int GlobalYROOTToCocoa(CGFloat yROOT);

int LocalYCocoaToROOT(NSView<X11Window> *parentView, CGFloat yCocoa);
int LocalYROOTToCocoa(NSView<X11Window> *parentView, CGFloat yROOT);
int LocalYROOTToCocoa(NSObject<X11Drawable> *parentView, CGFloat yROOT);

NSPoint TranslateToScreen(NSView<X11Window> *from, NSPoint point);
NSPoint TranslateFromScreen(NSPoint point, NSView<X11Window> *to);
NSPoint TranslateCoordinates(NSView<X11Window> *fromView, NSView<X11Window> *toView, NSPoint sourcePoint);

bool ViewIsTextViewFrame(NSView<X11Window> *view, bool checkParent);
bool LockFocus(NSView<X11Window> *view);
void UnlockFocus(NSView<X11Window> *view);//For symmetry only.

//Find intersection of view and sibling, result is a rect in view's space.
NSRect FindOverlapRect(const NSRect &viewRect, const NSRect &siblingViewRect);
bool RectsOverlap(const NSRect &r1, const NSRect &r2);

}//X11
}//MacOSX
}//ROOT

#endif
