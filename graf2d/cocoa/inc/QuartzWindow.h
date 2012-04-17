//Author: Timur Pocheptsov 16/02/2012

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


@interface QuartzWindow : NSWindow<X11Drawable>//, NSWindowDelegate>

@property (nonatomic, assign) QuartzPixmap *fBackBuffer;
@property (nonatomic, assign) QuartzView *fParentView;
@property (nonatomic, assign) unsigned fID;

/////////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t

@property (nonatomic, assign) long fEventMask;

@property (nonatomic, assign) int fClass;
@property (nonatomic, assign) int fDepth;

@property (nonatomic, assign) int fBitGravity;
@property (nonatomic, assign) int fWinGravity;

@property (nonatomic, assign) unsigned long fBackgroundPixel;

@property (nonatomic, readonly) int fMapState;


//End of SetWindowAttributes_t/WindowAttributes_t
/////////////////////////////////////////////////////////////////

@property (nonatomic, readonly) BOOL fIsPixmap;
@property (nonatomic, readonly) QuartzView *fContentView;
@property (nonatomic, readonly) QuartzWindow *fQuartzWindow;

@property (nonatomic, readonly) CGContextRef fContext;

@property (nonatomic, assign) int fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL fOwnerEvents;


//Life-cycle.

- (id) initWithContentRect : (NSRect) contentRect styleMask : (NSUInteger) windowStyle 
       backing : (NSBackingStoreType) bufferingType defer : (BOOL) deferCreation
       windowAttributes : (const SetWindowAttributes_t *) attr;

//Geometry.
- (int)      fX;
- (int)      fY;

- (unsigned) fWidth;
- (unsigned) fHeight;
- (NSSize)   fSize;

- (void)     setDrawableSize : (NSSize) newSize;
- (void)     setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void)     setX : (int) x Y : (int) y;

//Children subviews.
- (void)     addChild : (QuartzView *)child;

//X11/ROOT GUI's attributes.
- (void)     getAttributes : (WindowAttributes_t *) attr;
- (void)     setAttributes : (const SetWindowAttributes_t *) attr;

//
- (void)     mapRaised;
- (void)     mapWindow;
- (void)     mapSubwindows;

- (void)     unmapWindow;
//
- (void)     copy : (id<X11Drawable>) src area : (Rectangle_t) area withMask : (QuartzImage *)mask 
             clipOrigin : (Point_t) origin toPoint : (Point_t) dstPoint;

- (unsigned char *) readColorBits : (Rectangle_t) area;

@end

//////////////////////////////////////////////////////////////
//                                                          //
// I have to attach passive key grabs to a view.            //
//                                                          //
//////////////////////////////////////////////////////////////

@interface PassiveKeyGrab : NSObject
- (Int_t) fKeyCode;
- (UInt_t) fModifiers;
- (id) initWithKey : (Int_t) keyCode modifiers : (UInt_t) modifiers;
- (BOOL) matchKey : (Int_t) keyCode modifiers : (UInt_t) modifiers;
@end

////////////////////////////////////////
//                                    //
// QuartzView class - child window.   //
//                                    //
////////////////////////////////////////

@interface QuartzView : NSView<X11Drawable>

@property (nonatomic, assign) QuartzPixmap *fBackBuffer;
@property (nonatomic, assign) QuartzView *fParentView;

@property (nonatomic, assign) unsigned fID;
@property (nonatomic, assign) unsigned fLevel;
@property (nonatomic, readonly) BOOL fIsOverlapped;

/////////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t

@property (nonatomic, assign) long fEventMask;

@property (nonatomic, assign) int fClass;
@property (nonatomic, assign) int fDepth;

@property (nonatomic, assign) int fBitGravity;
@property (nonatomic, assign) int fWinGravity;

@property (nonatomic, assign) unsigned long fBackgroundPixel;

@property (nonatomic, readonly) int fMapState;

//End of SetWindowAttributes_t/WindowAttributes_t
/////////////////////////////////////////////////////////////////

@property (nonatomic, readonly) BOOL fIsPixmap;
@property (nonatomic, readonly) QuartzView *fContentView;
@property (nonatomic, readonly) QuartzWindow *fQuartzWindow;

@property (nonatomic, assign) CGContextRef fContext;

@property (nonatomic, assign) int fGrabButton;
@property (nonatomic, assign) unsigned fGrabButtonEventMask;
@property (nonatomic, assign) unsigned fGrabKeyModifiers;
@property (nonatomic, assign) BOOL fOwnerEvents;
//modifier also.


//Life-cycle.
- (id) initWithFrame : (NSRect) frame windowAttributes : (const SetWindowAttributes_t *) attr;

//Geometry.

- (BOOL)     isFlipped;//override method from NSView.

- (int)      fX;
- (int)      fY;

- (unsigned) fWidth;
- (unsigned) fHeight;
- (NSSize)   fSize;

- (void)     setDrawableSize : (NSSize) newSize;
- (void)     setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h;
- (void)     setX : (int) x Y : (int) y;

//Children subviews.
- (void)     addChild : (QuartzView *)child;

//X11/ROOT GUI's attributes.
- (void)     getAttributes : (WindowAttributes_t *)attr;
- (void)     setAttributes : (const SetWindowAttributes_t *)attr;

//
- (void)     mapRaised;
- (void)     mapWindow;
- (void)     mapSubwindows;

- (void)     unmapWindow;
//
- (void)     raiseWindow;
- (void)     lowerWindow;
//
- (void)     copy : (id<X11Drawable>) src area : (Rectangle_t) area withMask : (QuartzImage *)mask 
             clipOrigin : (Point_t) origin toPoint : (Point_t) dstPoint;

- (unsigned char *) readColorBits : (Rectangle_t) area;

//
- (void)     configureNotifyTree;
- (void)     updateLevel : (unsigned) newLevel;

//Keyboard:
- (void)     addPassiveKeyGrab : (Int_t) keyCode modifiers : (UInt_t) modifiers;
- (void)     removePassiveKeyGrab : (Int_t) keyCode modifiers : (UInt_t) modifiers;
- (PassiveKeyGrab *) findPassiveKeyGrab : (Int_t) keyCode modifiers : (UInt_t) modifiers;

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

//Coordinate conversion.
int GlobalYCocoaToROOT(CGFloat yCocoa);
int GlobalYROOTToCocoa(CGFloat yROOT);

int LocalYCocoaToROOT(QuartzView *parentView, CGFloat yCocoa);
int LocalYROOTToCocoa(QuartzView *parentView, CGFloat yROOT);
int LocalYROOTToCocoa(NSObject<X11Drawable> *parentView, CGFloat yROOT);

NSPoint TranslateToScreen(QuartzView *from, NSPoint point);
NSPoint TranslateFromScreen(NSPoint point, QuartzView *to);
NSPoint TranslateCoordinates(QuartzView *fromView, QuartzView *toView, NSPoint sourcePoint);

}//X11
}//MacOSX
}//ROOT

#endif
