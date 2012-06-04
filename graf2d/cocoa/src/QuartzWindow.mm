// @(#)root/graf2d:$Id$
// Author: Timur Pocheptsov   16/02/2012

/*************************************************************************
 * Copyright (C) 1995-2012, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//#define DEBUG_ROOT_COCOA

//#define NDEBUG

#ifdef DEBUG_ROOT_COCOA
#import <iostream>
#import <fstream>

#import "TClass.h"
#endif

#import <algorithm>
#import <stdexcept>
#import <cassert>

#import "QuartzWindow.h"
#import "QuartzPixmap.h"
#import "X11Buffer.h"
#import "X11Events.h"
#import "TGWindow.h"
#import "TGClient.h"
#import "TSystem.h"
#import "TGCocoa.h"

namespace ROOT {
namespace MacOSX {
namespace X11 {

//______________________________________________________________________________
QuartzWindow *CreateTopLevelWindow(Int_t x, Int_t y, UInt_t w, UInt_t h, UInt_t /*border*/, Int_t depth,
                                   UInt_t clss, void */*visual*/, SetWindowAttributes_t *attr, UInt_t)
{
   NSRect winRect = {};
   winRect.origin.x = x; 
   winRect.origin.y = GlobalYROOTToCocoa(y);
   winRect.size.width = w;
   winRect.size.height = h;

   //TODO check mask.
   const NSUInteger styleMask = NSTitledWindowMask | NSClosableWindowMask | NSMiniaturizableWindowMask | NSResizableWindowMask;
   //
   QuartzWindow *newWindow = [[QuartzWindow alloc] initWithContentRect : winRect styleMask : styleMask backing : NSBackingStoreBuffered defer : YES windowAttributes : attr];
   if (!newWindow)
      throw std::runtime_error("CreateTopLevelWindow failed");
   //
   newWindow.fDepth = depth;
   newWindow.fClass = clss;

   return newWindow;
}

//______________________________________________________________________________
QuartzView *CreateChildView(QuartzView * /*parent*/, Int_t x, Int_t y, UInt_t w, UInt_t h, UInt_t /*border*/, Int_t /*depth*/,
                            UInt_t /*clss*/, void * /*visual*/, SetWindowAttributes_t *attr, UInt_t /*wtype*/)
{
   NSRect viewRect = {};
   viewRect.origin.x = x;
   viewRect.origin.y = y;
   viewRect.size.width = w;
   viewRect.size.height = h;
   
   QuartzView *view = [[QuartzView alloc] initWithFrame : viewRect windowAttributes : attr];
   if (!view)
      throw std::runtime_error("CreateChildView failed");
   
   return view;
}

//______________________________________________________________________________
void GetRootWindowAttributes(WindowAttributes_t *attr)
{
   //'root' window does not exist, but we can request its attributes.
   assert(attr != 0 && "GetRootWindowAttributes, attr parameter is null");
   
   NSArray *screens = [NSScreen screens];
   assert(screens != nil && "screens array is nil");
   
   NSScreen *mainScreen = [screens objectAtIndex : 0];
   assert(mainScreen != nil && "screen with index 0 is nil");

   *attr = WindowAttributes_t();
   
   attr->fX = 0;
   attr->fY = 0;
   
   const NSRect frame = [mainScreen frame];
   
   attr->fWidth = frame.size.width;
   attr->fHeight = frame.size.height;
   attr->fBorderWidth = 0;
   attr->fYourEventMask = 0;
   attr->fAllEventMasks = 0;//???

   attr->fDepth = NSBitsPerPixelFromDepth([mainScreen depth]);
   attr->fVisual = 0;
   attr->fRoot = 0;
}


//Coordinate conversion.

//TODO: check how TGX11 extracts/changes window attributes.

//______________________________________________________________________________
int GlobalYCocoaToROOT(CGFloat yCocoa)
{
   NSArray *screens = [NSScreen screens];
   assert(screens != nil && "GlobalYCocoaToROOT, screens array is nil");
   
   NSScreen *mainScreen = [screens objectAtIndex : 0];
   assert(mainScreen != nil && "GlobalYCocoaToROOT, screen at index 0 is nil");
   
   return int(mainScreen.frame.size.height - yCocoa);
}

//______________________________________________________________________________
int GlobalYROOTToCocoa(CGFloat yROOT)
{
   //hehe :)) actually, no need in this function.
   NSArray *screens = [NSScreen screens];
   assert(screens != nil && "GlobalYROOTToCocoa, screens array is nil");
   
   NSScreen *mainScreen = [screens objectAtIndex : 0];
   assert(mainScreen != nil && "GlobalYROOTToCocoa, screen at index 0 is nil");
   
   return int(mainScreen.frame.size.height - yROOT);
}

//______________________________________________________________________________
int LocalYCocoaToROOT(NSView<X11Window> *parentView, CGFloat yCocoa)
{
   assert(parentView != nil && "LocalYCocoaToROOT, parent view is nil");
   
   return int(parentView.frame.size.height - yCocoa);
}

//______________________________________________________________________________
int LocalYROOTToCocoa(NSView<X11Window> *parentView, CGFloat yROOT)
{
   //:)
   assert(parentView != nil && "LocalYROOTToCocoa, parent view is nil");
   
   return int(parentView.frame.size.height - yROOT);
}


//______________________________________________________________________________
int LocalYROOTToCocoa(NSObject<X11Drawable> *drawable, CGFloat yROOT)
{
   //:)
   assert(drawable != nil && "LocalYROOTToCocoa, drawable is nil");
   
   return int(drawable.fHeight - yROOT);
}

//______________________________________________________________________________
NSPoint TranslateToScreen(NSView<X11Window> *from, NSPoint point)
{
   assert(from != nil && "TranslateToScreen, 'from' parameter is nil");
   
   //TODO: I do not know, if I can use convertToBacking ..... - have to check this.
   NSPoint winPoint = [from convertPoint : point toView : nil];
   NSPoint screenPoint = [[from window] convertBaseToScreen : winPoint];; 
   //TODO: This is Cocoa's coordinates, but for ROOT I have to convert.
   screenPoint.y = GlobalYCocoaToROOT(screenPoint.y);

   return screenPoint;
}

//______________________________________________________________________________
NSPoint TranslateFromScreen(NSPoint point, NSView<X11Window> *to)
{
   assert(to != nil && "TranslateFromScreen, 'to' parameter is nil");
   
   point.y = GlobalYROOTToCocoa(point.y);

   //May be I can use convertBackingTo .... have to check this.
   const NSPoint winPoint = [[to window] convertScreenToBase : point];
   return [to convertPoint : winPoint fromView : nil];
}

//______________________________________________________________________________
NSPoint TranslateCoordinates(NSView<X11Window> *from, NSView<X11Window> *to, NSPoint sourcePoint)
{
   //Both views are valid.
   assert(from != nil && "TranslateCoordinates, 'from' parameter is nil");
   assert(to != nil && "TranslateCoordinates, 'to' parameter is nil");

   if ([from window] == [to window]) {
      //Both views are in the same window.
      return [to convertPoint : sourcePoint fromView : from];      
   } else {
      //May be, I can do it in one call, but it's not obvious for me
      //what is 'pixel aligned backing store coordinates' and
      //if they are the same as screen coordinates.
      
      const NSPoint win1Point = [from convertPoint : sourcePoint toView : nil];
      const NSPoint screenPoint = [[from window] convertBaseToScreen : win1Point];
      const NSPoint win2Point = [[to window] convertScreenToBase : screenPoint];

      return [to convertPoint : win2Point fromView : nil];
   }
}

//______________________________________________________________________________
void SetWindowAttributes(const SetWindowAttributes_t *attr, NSObject<X11Window> *window)
{
   assert(attr != 0 && "SetWindowAttributes, attr parameter is null");
   assert(window != nil && "SetWindowAttributes, window parameter is nil");

   const Mask_t mask = attr->fMask;

   if (mask & kWABackPixel)
      window.fBackgroundPixel = attr->fBackgroundPixel;
   
   if (mask & kWAEventMask)
      window.fEventMask = attr->fEventMask;

   if (mask & kWABitGravity)
      window.fBitGravity = attr->fBitGravity;

   if (mask & kWAWinGravity)
      window.fWinGravity = attr->fWinGravity;
   
   //TODO: More attributes to set -
   //cursor for example, etc.
   if (mask & kWAOverrideRedirect) {
      //This is quite a special case.
      //TODO: Must be checked yet, if I understand this correctly!
      if ([(NSObject *)window isKindOfClass : [QuartzWindow class]]) {
         QuartzWindow *qw = (QuartzWindow *)window;
         [qw setStyleMask : NSBorderlessWindowMask];
      }
   }
}

//______________________________________________________________________________
void GetWindowGeometry(NSObject<X11Window> *win, WindowAttributes_t *dst)
{
   assert(win != nil && "GetWindowGeometry, win parameter is nil");
   assert(dst != 0 && "GetWindowGeometry, dst paremeter is null");
   
   dst->fX = win.fX;
   dst->fY = win.fY;
   
   dst->fWidth = win.fWidth;
   dst->fHeight = win.fHeight;
}

//______________________________________________________________________________
void GetWindowAttributes(NSObject<X11Window> *window, WindowAttributes_t *dst)
{
   assert(window != nil && "GetWindowAttributes, window parameter is nil");
   assert(dst != 0 && "GetWindowAttributes, attr parameter is null");
   
   *dst = WindowAttributes_t();
   
   //fX, fY, fWidth, fHeight.
   GetWindowGeometry(window, dst);

   //Actually, most of them are not used by GUI.
   dst->fBorderWidth = 0;
   dst->fDepth = window.fDepth;
   //Dummy value.
   dst->fVisual = 0;
   //Dummy value.
   dst->fRoot = 0;
   dst->fClass = window.fClass;
   dst->fBitGravity = window.fBitGravity;
   dst->fWinGravity = window.fWinGravity;
   //Dummy value.
   dst->fBackingStore = kAlways;//??? CHECK
   dst->fBackingPlanes = 0;

   //Dummy value.
   dst->fBackingPixel = 0;
   
   dst->fSaveUnder = 0;

   //Dummy value.
   dst->fColormap = 0;
   //Dummy value.   
   dst->fMapInstalled = kTRUE;

   dst->fMapState = window.fMapState;

   dst->fAllEventMasks = window.fEventMask;
   dst->fYourEventMask = window.fEventMask;
   
   //Not used by GUI.
   //dst->fDoNotPropagateMask

   dst->fOverrideRedirect = 0;
   //Dummy value.
   dst->fScreen = 0;
}

//With Apple's poor man's objective C/C++ + "brilliant" Cocoa you never know, what should be 
//the linkage of callback functions, API + language dialects == MESS. I declare/define this comparators here
//as having "C++" linkage. If one good day clang will start to complane, I'll have to change this.

//______________________________________________________________________________
NSComparisonResult CompareViewsToLower(id view1, id view2, void *context)
{
    id topView = (id)context;//ARC will require _brigde cast, but NO ARC! :)
    if (view1 == topView)
        return NSOrderedAscending;
    if (view2 == topView)
        return NSOrderedDescending;
    return NSOrderedSame;
}

//______________________________________________________________________________
NSComparisonResult CompareViewsToRaise(id view1, id view2, void *context)
{
   id topView = (id)context;
   if (view1 == topView)
      return NSOrderedDescending;
   if (view2 == topView)
      return NSOrderedAscending;

   return NSOrderedSame;
}

//______________________________________________________________________________
NSPoint GetCursorHotStop(NSImage *image, ECursor cursor)
{
   assert(image != nil && "CursroHotSpot, image parameter is nil");
   
   const NSSize imageSize = image.size;

   if (cursor == kArrowRight) 
      return CGPointMake(imageSize.width, imageSize.height / 2);
   
   return CGPointMake(imageSize.width / 2, imageSize.height / 2);
}

//TGTextView is a very special window: it's a TGCompositeFrame,
//which has TGCompositeFrame inside (TGViewFrame). This TGViewFrame
//delegates Expose events to its parent, and parent tries to draw
//inside a TGViewFrame. This does not work with default 
//QuartzView -drawRect/TGCocoa. So I need a trick to identify
//this special window.

//______________________________________________________________________________
bool ViewIsTextView(unsigned viewID)
{
   TGWindow *window = gClient->GetWindowById(viewID);
   if (!window)
      return false;   
   return window->InheritsFrom("TGTextView");
}

//______________________________________________________________________________
bool ViewIsTextView(NSView<X11Window> *view)
{
   assert(view != nil && "ViewIsTextView, view parameter is nil");

   return ViewIsTextView(view.fID);
}

//______________________________________________________________________________
bool ViewIsTextViewFrame(NSView<X11Window> *view, bool checkParent)
{
   assert(view != nil && "ViewIsTextViewFrame, view parameter is nil");
   
   TGWindow *window = gClient->GetWindowById(view.fID);
   if (!window)
      return false;

   if (!window->InheritsFrom("TGViewFrame"))
      return false;
      
   if (!checkParent)
      return true;
      
   if (!view.fParentView)
      return false;
      
   return ViewIsTextView(view.fParentView);
}

//______________________________________________________________________________
NSView<X11Window> *FrameForTextView(NSView<X11Window> *textView)
{
   assert(textView != nil && "FrameForTextView, textView parameter is nil");
   
   for (NSView<X11Window> *child in [textView subviews]) {
      if (ViewIsTextViewFrame(child, false))
         return child;
   }
   
   return nil;
}

//______________________________________________________________________________
bool LockFocus(NSView<X11Window> *view)
{
   assert(view != nil && "LockFocus, view parameter is nil");
   assert([view isKindOfClass : [QuartzView class]] && "LockFocus, QuartzView is expected");
   
   if ([view lockFocusIfCanDraw]) {
      NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
      assert(nsContext != nil && "LockFocus, currentContext is nil");
      CGContextRef currContext = (CGContextRef)[nsContext graphicsPort];
      assert(currContext != 0 && "LockFocus, graphicsPort is null");//remove this assert?
      
      ((QuartzView *)view).fContext = currContext;
      
      return true;
   }
   
   return false;
}

//______________________________________________________________________________
void UnlockFocus(NSView<X11Window> *view)
{
   assert(view != nil && "UnlockFocus, view parameter is nil");
   assert([view isKindOfClass : [QuartzView class]] && "UnlockFocus, QuartzView is expected");
   
   [view unlockFocus];
   ((QuartzView *)view).fContext = 0;
}

//________________________________________________________________________________________
NSRect FindOverlapRect(const NSRect &viewRect, const NSRect &siblingViewRect)
{
   NSRect frame1 = viewRect;
   NSRect frame2 = siblingViewRect;

   //Adjust frames - move to frame1's space.
   frame2.origin.x -= frame1.origin.x;
   frame2.origin.y -= frame1.origin.y;
   frame1.origin = CGPointZero;

   NSRect overlap = {};
   
   if (frame2.origin.x < 0) {
      overlap.size.width = std::min(frame1.size.width, frame2.size.width - (frame1.origin.x - frame2.origin.x));
   } else {
      overlap.origin.x = frame2.origin.x;
      overlap.size.width = std::min(frame2.size.width, frame1.size.width - frame2.origin.x);
   }
   
   if (frame2.origin.y < 0) {
      overlap.size.height = std::min(frame1.size.height, frame2.size.height - (frame1.origin.y - frame2.origin.y));
   } else {
      overlap.origin.y = frame2.origin.y;
      overlap.size.height = std::min(frame2.size.height, frame1.size.height - frame2.origin.y);
   }
   
   return overlap;

}

//________________________________________________________________________________________
bool RectsOverlap(const NSRect &r1, const NSRect &r2)
{
   if (r2.origin.x >= r1.origin.x + r1.size.width)
      return false;
   if (r2.origin.x + r2.size.width <= r1.origin.x)
      return false;
   if (r2.origin.y >= r1.origin.y + r1.size.height)
      return false;
   if (r2.origin.y + r2.size.height <= r1.origin.y)
      return false;
   
   return true;
}

}//X11
}//MacOSX
}//ROOT

#ifdef DEBUG_ROOT_COCOA

namespace {

//______________________________________________________________________________
void log_attributes(const SetWindowAttributes_t *attr, unsigned winID)
{
   //This function is loggin requests, at the moment I can not set all
   //of these attributes, so I first have to check, what is actually
   //requested by ROOT.
   static std::ofstream logfile("win_attr.txt");

   const Mask_t mask = attr->fMask;   
   if (mask & kWABackPixmap)
      logfile<<"win "<<winID<<": BackPixmap\n";
   if (mask & kWABackPixel)
      logfile<<"win "<<winID<<": BackPixel\n";
   if (mask & kWABorderPixmap)
      logfile<<"win "<<winID<<": BorderPixmap\n";
   if (mask & kWABorderPixel)
      logfile<<"win "<<winID<<": BorderPixel\n";
   if (mask & kWABorderWidth)
      logfile<<"win "<<winID<<": BorderWidth\n";
   if (mask & kWABitGravity)
      logfile<<"win "<<winID<<": BitGravity\n";
   if (mask & kWAWinGravity)
      logfile<<"win "<<winID<<": WinGravity\n";
   if (mask & kWABackingStore)
      logfile<<"win "<<winID<<": BackingStore\n";
   if (mask & kWABackingPlanes)
      logfile<<"win "<<winID<<": BackingPlanes\n";
   if (mask & kWABackingPixel)
      logfile<<"win "<<winID<<": BackingPixel\n";
   if (mask & kWAOverrideRedirect)
      logfile<<"win "<<winID<<": OverrideRedirect\n";
   if (mask & kWASaveUnder)
      logfile<<"win "<<winID<<": SaveUnder\n";
   if (mask & kWAEventMask)
      logfile<<"win "<<winID<<": EventMask\n";
   if (mask & kWADontPropagate)
      logfile<<"win "<<winID<<": DontPropagate\n";
   if (mask & kWAColormap)
      logfile<<"win "<<winID<<": Colormap\n";
   if (mask & kWACursor)
      logfile<<"win "<<winID<<": Cursor\n";
}

//______________________________________________________________________________
void print_mask_info(ULong_t mask)
{
   if (mask & kButtonPressMask)
      NSLog(@"button press mask");
   if (mask & kButtonReleaseMask)
      NSLog(@"button release mask");
   if (mask & kExposureMask)
      NSLog(@"exposure mask");
   if (mask & kPointerMotionMask)
      NSLog(@"pointer motion mask");
   if (mask & kButtonMotionMask)
      NSLog(@"button motion mask");
   if (mask & kEnterWindowMask)
      NSLog(@"enter notify mask");
   if (mask & kLeaveWindowMask)
      NSLog(@"leave notify mask");
}

}
#endif


@implementation QuartzWindow {
@private
   QuartzView *fContentView;
   BOOL fDelayedTransient;
}


@synthesize fMainWindow;
@synthesize fBackBuffer;

//QuartzWindow's life cycle.
//______________________________________________________________________________
- (id) initWithContentRect : (NSRect) contentRect styleMask : (NSUInteger) windowStyle backing : (NSBackingStoreType) bufferingType 
       defer : (BOOL) deferCreation  windowAttributes : (const SetWindowAttributes_t *)attr
{
   self = [super initWithContentRect : contentRect styleMask : windowStyle backing : bufferingType defer : deferCreation];

   if (self) {
      //ROOT's not able to draw GUI concurrently, thanks to global variables and gVirtualX itself.
      [self setAllowsConcurrentViewDrawing : NO];

      self.delegate = self;
      //create content view here.
      NSRect contentViewRect = contentRect;
      contentViewRect.origin.x = 0.f;
      contentViewRect.origin.y = 0.f;

      //TODO: OpenGL view can not be content of our QuartzWindow, check if
      //this is a problem for ROOT.
      fContentView = [[QuartzView alloc] initWithFrame : contentViewRect windowAttributes : 0];
      
      [self setContentView : fContentView];

      [fContentView release];
      fDelayedTransient = NO;
      
      if (attr)
         ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
   }
   
   return self;
}

//______________________________________________________________________________
- (void) addTransientWindow : (QuartzWindow *)window
{
   assert(window != nil && "addTransientWindow, window parameter is nil");

   window.fMainWindow = self;
   
   if (window.fMapState != kIsViewable) {
      window.fDelayedTransient = YES;
   } else {
      [self addChildWindow : window ordered : NSWindowAbove];
      window.fDelayedTransient = NO;
   }
}

//______________________________________________________________________________
- (void) dealloc
{
   [super dealloc];
}

//______________________________________________________________________________
- (void) setFDelayedTransient : (BOOL) d
{
   fDelayedTransient = d;
}

///////////////////////////////////////////////////////////
//X11Drawable's protocol.

//______________________________________________________________________________
- (unsigned) fID 
{
   assert(fContentView != nil && "fID, content view is nil");

   return fContentView.fID;
}

//______________________________________________________________________________
- (void) setFID : (unsigned) winID
{
   assert(fContentView != nil && "setFID, content view is nil");
   
   fContentView.fID = winID;
}

//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return NO;
}

//______________________________________________________________________________
- (BOOL) fIsOpenGLWidget
{
   return NO;
}

//______________________________________________________________________________
- (CGContextRef) fContext 
{
   assert(fContentView != nil && "fContext, fContentView is nil");

   return fContentView.fContext;
}

//______________________________________________________________________________
- (void) setFContext : (CGContextRef) ctx
{
   assert(fContentView != nil && "setFContext, fContentView is nil");

   fContentView.fContext = ctx;
}

//______________________________________________________________________________
- (int) fX
{
   return self.frame.origin.x;
}

//______________________________________________________________________________
- (int) fY
{
   return ROOT::MacOSX::X11::GlobalYCocoaToROOT(self.frame.origin.y + self.frame.size.height);
}

//______________________________________________________________________________
- (unsigned) fWidth
{
   return self.frame.size.width;
}

//______________________________________________________________________________
- (unsigned) fHeight
{
   //NSWindow's frame (height component) also includes title-bar.
   //So I have to use content view's height.
   //Obviously, there is a "hole" == 22 pixels.
   assert(fContentView != nil && "fHeight, content view is nil");
   
   return fContentView.frame.size.height;
}

//______________________________________________________________________________
- (void) setDrawableSize : (NSSize) newSize
{
   //Can not simply do self.frame.size = newSize.
   assert(!(newSize.width < 0) && "setDrawableSize, width is negative");
   assert(!(newSize.height < 0) && "setDrawableSize, height is negative");
   
   [self setContentSize : newSize];
}

//______________________________________________________________________________
- (void) setX : (int) x Y : (int) y width : (unsigned) w height : (unsigned) h
{
   NSSize newSize = {};
   newSize.width = w;
   newSize.height = h;
   [self setContentSize : newSize];
   
   //Check how this is affected by title bar's height.
   NSPoint topLeft = {};
   topLeft.x = x;
   topLeft.y = ROOT::MacOSX::X11::GlobalYROOTToCocoa(y);

   [self setFrameTopLeftPoint : topLeft];
}

//______________________________________________________________________________
- (void) setX : (int) x Y : (int) y
{
   NSPoint topLeft = {};
   topLeft.x = x;
   topLeft.y = ROOT::MacOSX::X11::GlobalYROOTToCocoa(y);

   [self setFrameTopLeftPoint : topLeft];
}

//______________________________________________________________________________
- (void) copy : (NSObject<X11Drawable> *) src area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   assert(fContentView != nil && "copy:area:toPoint:, fContentView is nil");

   [fContentView copy : src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
}

//______________________________________________________________________________
- (unsigned char *) readColorBits : (Rectangle_t) area
{
   assert(fContentView != nil && "readColorBits:, fContentView is nil");
   
   return [fContentView readColorBits : area];
}

//X11Window protocol's implementation.

//______________________________________________________________________________
- (QuartzView *) fParentView
{
   return nil;
}

//______________________________________________________________________________
- (void) setFParentView : (QuartzView *) parent
{
   (void)parent;
}

//______________________________________________________________________________
- (NSView<X11Window> *) fContentView
{
   return fContentView;
}

//______________________________________________________________________________
- (QuartzWindow *) fQuartzWindow
{
   return self;
}

/////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t.

//______________________________________________________________________________
- (long) fEventMask
{
   assert(fContentView != nil && "fEventMask, content view is nil");
   
   return fContentView.fEventMask;
}

//______________________________________________________________________________
- (void) setFEventMask : (long)mask 
{
   assert(fContentView != nil && "setFEventMask, content view is nil");
   
   fContentView.fEventMask = mask;
}

//______________________________________________________________________________
- (int) fClass
{
   assert(fContentView != nil && "fClass, content view is nil");
   
   return fContentView.fClass;
}

//______________________________________________________________________________
- (void) setFClass : (int) windowClass
{
   assert(fContentView != nil && "setFClass, content view is nil");
   
   fContentView.fClass = windowClass;
}

//______________________________________________________________________________
- (int) fDepth
{
   assert(fContentView != nil && "fDepth, content view is nil");
   
   return fContentView.fDepth;
}

//______________________________________________________________________________
- (void) setFDepth : (int) depth
{
   assert(fContentView != nil && "setFDepth, content view is nil");
   
   fContentView.fDepth = depth;
}

//______________________________________________________________________________
- (int) fBitGravity
{
   assert(fContentView != nil && "fBitGravity, content view is nil");
   
   return fContentView.fBitGravity;
}

//______________________________________________________________________________
- (void) setFBitGravity : (int) bit
{
   assert(fContentView != nil && "setFBitGravity, content view is nil");

   fContentView.fBitGravity = bit;
}

//______________________________________________________________________________
- (int) fWinGravity
{
   assert(fContentView != nil && "fWinGravity, content view is nil");
   
   return fContentView.fWinGravity;
}

//______________________________________________________________________________
- (void) setFWinGravity : (int) bit
{
   assert(fContentView != nil && "setFWinGravity, content view is nil");
   
   fContentView.fWinGravity = bit;
}

//______________________________________________________________________________
- (unsigned long) fBackgroundPixel
{
   assert(fContentView != nil && "fBackgroundPixel, content view is nil");
   
   return fContentView.fBackgroundPixel;
}

//______________________________________________________________________________
- (void) setFBackgroundPixel : (unsigned long) pixel
{
   assert(fContentView != nil && "SetFBackgroundPixel, content view is nil");
   
   fContentView.fBackgroundPixel = pixel;
}

//______________________________________________________________________________
- (int) fMapState
{
   //Top-level window can be only kIsViewable or kIsUnmapped (not unviewable).
   assert(fContentView != nil && "fMapState, content view is nil");
   
   if ([fContentView isHidden])
      return kIsUnmapped;
      
   return kIsViewable;
}

//______________________________________________________________________________
- (int) fGrabButton
{
   assert(fContentView != nil && "fGrabButton, content view is nil");
   
   return fContentView.fGrabButton;
}

//______________________________________________________________________________
- (void) setFGrabButton : (int) btn
{
   assert(fContentView != nil && "setFGrabButton, content view is nil");
   
   fContentView.fGrabButton = btn;
}

//______________________________________________________________________________
- (unsigned) fGrabButtonEventMask
{
   assert(fContentView != nil && "fGrabButtonEventMask, content view is nil");
   
   return fContentView.fGrabButtonEventMask;
}

//______________________________________________________________________________
- (void) setFGrabButtonEventMask : (unsigned) mask
{
   assert(fContentView != nil && "setFGrabButtonEventMask, content view is nil");
   
   fContentView.fGrabButtonEventMask = mask;
}

//______________________________________________________________________________
- (unsigned) fGrabKeyModifiers
{
   assert(fContentView != nil && "fGrabKeyModifiers, content view is nil");
   
   return fContentView.fGrabKeyModifiers;
}

//______________________________________________________________________________
- (void) setFGrabKeyModifiers : (unsigned) mod
{
   assert(fContentView != nil && "setFGrabKeyModifiers, content view is nil");
   
   fContentView.fGrabKeyModifiers = mod;
}

//______________________________________________________________________________
- (BOOL) fOwnerEvents
{
   assert(fContentView != nil && "fOwnerEvents, content view is nil");

   return fContentView.fOwnerEvents;
}

//______________________________________________________________________________
- (void) setFOwnerEvents : (BOOL) owner
{
   assert(fContentView != nil && "setFOwnerEvents, content view is nil");

   fContentView.fOwnerEvents = owner;
}


//______________________________________________________________________________
- (void) addChild : (NSView<X11Window> *) child
{
   assert(child != nil && "addChild, child view is nil");
 
   if (!fContentView) {
      //This can happen only in case of re-parent operation.
      assert([child isKindOfClass : [QuartzView class]] && "addChild: gl view in a top-level window as content view is not supported");
      fContentView = (QuartzView *)child;
      [self setContentView : child];
      fContentView.fParentView = nil;
   } else
      [fContentView addChild : child];
}

//______________________________________________________________________________
- (void) getAttributes : (WindowAttributes_t *) attr
{
   assert(fContentView != 0 && "getAttributes, content view is nil");
   assert(attr && "getAttributes, attr parameter is nil");

   ROOT::MacOSX::X11::GetWindowAttributes(self, attr);
}

//______________________________________________________________________________
- (void) setAttributes : (const SetWindowAttributes_t *)attr
{
   assert(attr != 0 && "setAttributes, attr parameter is null");

#ifdef DEBUG_ROOT_COCOA
   log_attributes(attr, self.fID);
#endif

   ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
}

//______________________________________________________________________________
- (void) mapRaised
{
   assert(fContentView && "mapRaised, content view is nil");

//   [self orderFront : self];
   [self makeKeyAndOrderFront : self];
   [fContentView setHidden : NO];
   [fContentView configureNotifyTree];

   if (fDelayedTransient) {
      fDelayedTransient = NO;
      [fMainWindow addChildWindow : self ordered : NSWindowAbove];
   }
}

//______________________________________________________________________________
- (void) mapWindow
{
   assert(fContentView != nil && "mapWindow, content view is nil");

//   [self orderFront : self];
   [self makeKeyAndOrderFront : self];
   [fContentView setHidden : NO];
   [fContentView configureNotifyTree];
   
   if (fDelayedTransient) {
      fDelayedTransient = NO;
      [fMainWindow addChildWindow : self ordered : NSWindowAbove];
   }
}

//______________________________________________________________________________
- (void) mapSubwindows
{
   assert(fContentView != nil && "mapSubwindows, content view is nil");

   [fContentView mapSubwindows];
   [fContentView configureNotifyTree];
}

//______________________________________________________________________________
- (void) unmapWindow
{
   assert(fContentView != nil && "unmapWindow, content view is nil");

   [fContentView setHidden : YES];
   [self orderOut : self];
}

//Events.

//______________________________________________________________________________
- (void) sendEvent : (NSEvent *) theEvent
{
   assert(fContentView != nil && "sendEvent, content view is nil");

   if (theEvent.type == NSLeftMouseDown || theEvent.type == NSRightMouseDown) {
      const NSPoint windowPoint = [theEvent locationInWindow];
      const NSPoint viewPoint =  [fContentView convertPointFromBase : windowPoint];
      if (viewPoint.y <= 0 && windowPoint.y >= 0) {
         //Very special case: mouse is in a title bar. 
         //There are not NSView<X11Window> object in this area,
         //this event will never go to any QuartzView, and this
         //can be a problem: if drop-down menu is open and
         //you move window using title-bar, ROOT's menu will
         //"fell through" the main window, which is weird.
         TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
         assert(vx != 0 && "sendEvent, gVirtualX is either null or not of TGCocoa type");
         if (vx->GetEventTranslator()->HasPointerGrab())
            vx->GetEventTranslator()->GenerateButtonReleaseEvent(fContentView, theEvent, theEvent.type == NSLeftMouseDown ? kButton1 : kButton3);//yes, button release???
      }
   }

   //Always call default version.
   [super sendEvent : theEvent];
}

//Cursors.
//______________________________________________________________________________
- (ECursor) fCurrentCursor
{
   assert(fContentView != nil && "fCurrentCursor, content view is nil");
   
   return fContentView.fCurrentCursor;
}

//______________________________________________________________________________
- (void) setFCurrentCursor : (ECursor) cursor
{
   assert(fContentView != nil && "setFCurrentCursor, content view is nil");
   
   fContentView.fCurrentCursor = cursor;
}


//NSWindowDelegate's methods.

//______________________________________________________________________________
- (BOOL) windowShouldClose : (id) sender
{
   (void)sender;

   assert(fContentView != nil && "windowShouldClose, content view is nil");

   //TODO: check this!!! Children are
   //transient windows and ROOT does not handle
   //such a deletion properly, noop then.
   if ([[self childWindows] count])
      return NO;

   //Prepare client message for a window.
   Event_t closeEvent = {};
   closeEvent.fWindow = fContentView.fID;
   closeEvent.fType = kClientMessage;
   closeEvent.fFormat = 32;//Taken from GUI classes.
   closeEvent.fHandle = TGCocoa::kIA_DELETE_WINDOW;
   closeEvent.fUser[0] = TGCocoa::kIA_DELETE_WINDOW;
   //Place it into the queue.
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "windowShouldClose, gVirtualX is either null or has a type different from TGCocoa");
   vx->SendEvent(fContentView.fID, &closeEvent);

   //Do not let AppKit to close a window.
   return NO;
}

@end

//
//
//
//
//Passive key grab info.

@implementation PassiveKeyGrab {
   unichar fKeyCode;
   NSUInteger fModifiers;
}

//______________________________________________________________________________
- (id) initWithKey : (unichar) keyCode modifiers : (NSUInteger) modifiers
{
   if (self = [super init]) {
      fKeyCode = keyCode;
      fModifiers = modifiers;
   }
   
   return self;
}

//______________________________________________________________________________
- (BOOL) matchKey : (unichar) keyCode modifiers : (NSUInteger) modifiers
{
   return keyCode == fKeyCode && modifiers == fModifiers;
}

//______________________________________________________________________________
- (BOOL) matchKey : (unichar) keyCode
{
   return keyCode == fKeyCode;
}

//______________________________________________________________________________
- (unichar) fKeyCode 
{
   return fKeyCode;
}

//______________________________________________________________________________
- (NSUInteger) fModifiers
{
   return fModifiers;
}

@end


@implementation QuartzView {
   NSMutableArray *fPassiveKeyGrabs;
   BOOL            fIsOverlapped;
   QuartzImage    *fClipMask;
}

@synthesize fClipMaskIsValid;

@synthesize fID;
@synthesize fContext;
/////////////////////
//SetWindowAttributes_t/WindowAttributes_t
@synthesize fEventMask;
@synthesize fClass;
@synthesize fDepth;
@synthesize fBitGravity;
@synthesize fWinGravity;
@synthesize fBackgroundPixel;
//SetWindowAttributes_t/WindowAttributes_t
/////////////////////
@synthesize fBackBuffer;
@synthesize fParentView;
@synthesize fLevel;
@synthesize fGrabButton;
@synthesize fGrabButtonEventMask;
@synthesize fGrabKeyModifiers;
@synthesize fOwnerEvents;
@synthesize fSnapshotDraw;
@synthesize fCurrentCursor;

//______________________________________________________________________________
- (id) initWithFrame : (NSRect) frame windowAttributes : (const SetWindowAttributes_t *)attr
{
   if (self = [super initWithFrame : frame]) {
      //Make this explicit (though memory is zero initialized).
      fClipMaskIsValid = NO;
      fClipMask = nil;

      fID = 0;
      fLevel = 0;
      
      //Passive grab parameters.
      fGrabButton = -1;//0 is kAnyButton.
      fGrabButtonEventMask = 0;
      fOwnerEvents = NO;
      
      fPassiveKeyGrabs = [[NSMutableArray alloc] init];
      
      [self setCanDrawConcurrently : NO];
      
      [self setHidden : YES];
      //Actually, check if view need this.
      const NSUInteger trackerOptions = NSTrackingMouseMoved | NSTrackingMouseEnteredAndExited | NSTrackingActiveInActiveApp | NSTrackingInVisibleRect;
      frame.origin = CGPointZero;
      NSTrackingArea *tracker = [[NSTrackingArea alloc] initWithRect : frame options : trackerOptions owner : self userInfo : nil];
      [self addTrackingArea : tracker];
      [tracker release];
      //
      if (attr)
         ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
         
      fCurrentCursor = kPointer;
   }
   
   return self;
}

//Overlap management.
//______________________________________________________________________________
- (BOOL) initClipMask
{
   const NSSize size = self.frame.size;

   if (fClipMask) {
      if ((unsigned)size.width == fClipMask.fWidth && (unsigned)size.height == fClipMask.fHeight) {
         //All pixels must be visible.
         [fClipMask clearMask];
      } else {
         [fClipMask release];
         fClipMask = nil;
      }
   }
   
   if (!fClipMask) {
      fClipMask = [QuartzImage alloc];
      if ([fClipMask initMaskWithW : (unsigned)size.width H : (unsigned)size.height]) {
         return YES;
      } else {
         [fClipMask release];
         fClipMask = nil;
         return NO;
      }
   }

   return YES;
}

//______________________________________________________________________________
- (QuartzImage *) fClipMask
{
   return fClipMask;
}

//______________________________________________________________________________
- (void) addOverlap : (NSRect)overlapRect
{
   assert(fClipMask != nil && "addOverlap, fClipMask is nil");
   assert(fClipMaskIsValid == YES && "addOverlap, fClipMask is invalid");
   
   [fClipMask maskOutPixels : overlapRect];
}

//X11Drawable protocol.
//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return NO;
}

//______________________________________________________________________________
- (BOOL) fIsOpenGLWidget
{
   return NO;
}

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
   NSRect newFrame = self.frame;
   newFrame.origin.x = x;
   newFrame.origin.y = y;
   
   self.frame = newFrame;
}

//______________________________________________________________________________
- (void) copyImage : (QuartzImage *) srcImage area : (Rectangle_t) area withMask : (QuartzImage *) mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   //Check parameters.
   assert(srcImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage.fImage is nil");

   //Check self.
   assert(self.fContext != 0 && "copyImage:area:withMask:clipOrigin:toPoint:, self.fContext is null");
   
   CGImageRef subImage = 0;
   bool needSubImage = false;
   if (area.fX || area.fY || area.fWidth != srcImage.fWidth || area.fHeight != srcImage.fHeight) {
      needSubImage = true;
      subImage = ROOT::MacOSX::X11::CreateSubImage(srcImage, area);
      if (!subImage) {
         NSLog(@"QuartzView: -copyImage:area:withMask:clipOrigin:toPoint:, subimage creation failed");
         return;
      }
   } else
      subImage = srcImage.fImage;

   //Save context state.
   CGContextSaveGState(self.fContext);

   //Scale and translate to undo isFlipped.
   CGContextTranslateCTM(self.fContext, 0., self.fHeight); 
   CGContextScaleCTM(self.fContext, 1., -1.);
   //Set clip mask on a context.
   
   if (mask) {
      assert(mask.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) == true && "copyImage:area:withMask:clipOrigin:toPoint:, mask.fImage is not a mask");
      clipXY.fY = ROOT::MacOSX::X11::LocalYROOTToCocoa(self, clipXY.fY + mask.fHeight);
      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(self.fContext, clipRect, mask.fImage);
   }
   
   //Convert from X11 to Cocoa (as soon as we scaled y * -1).
   dstPoint.fY = ROOT::MacOSX::X11::LocalYROOTToCocoa(self, dstPoint.fY + area.fHeight);
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(self.fContext, imageRect, subImage);

   //Restore context state.
   CGContextRestoreGState(self.fContext);
   
   if (needSubImage)
      CGImageRelease(subImage);
}

//______________________________________________________________________________
- (void) copyView : (QuartzView *) srcView area : (Rectangle_t) area toPoint : (Point_t) dstPoint
{
   //To copy one "window" to another "window", I have to ask source QuartzView to draw intself into
   //bitmap, and copy this bitmap into the destination view.

   assert(srcView != nil && "copyView:area:toPoint:, srcView parameter is nil");

   const NSRect frame = [srcView frame];   
   NSBitmapImageRep *imageRep = [srcView bitmapImageRepForCachingDisplayInRect : frame];
   if (!imageRep) {
      NSLog(@"QuartzView: -copyView:area:toPoint failed");
      return;
   }
   
   assert(srcView != nil && "copyView:area:toPoint:, srcView parameter is nil");
   assert(self.fContext != 0 && "copyView:area:toPoint, self.fContext is null");

   //It can happen, that src and self are the same.
   //cacheDisplayInRect calls drawRect with bitmap context 
   //(and this will reset self.fContext: I have to save/restore it.
   CGContextRef ctx = srcView.fContext;
   srcView.fSnapshotDraw = YES;
   [srcView cacheDisplayInRect : frame toBitmapImageRep : imageRep];
   srcView.fSnapshotDraw = NO;
   srcView.fContext = ctx;

   const CGRect subImageRect = CGRectMake(area.fX, area.fY, area.fWidth, area.fHeight);
   CGImageRef subImage = CGImageCreateWithImageInRect(imageRep.CGImage, subImageRect);

   CGContextSaveGState(self.fContext);

   const CGRect imageRect = CGRectMake(dstPoint.fX, [self visibleRect].size.height - (dstPoint.fY + area.fHeight), area.fWidth, area.fHeight);

   CGContextTranslateCTM(self.fContext, 0., [self visibleRect].size.height); 
   CGContextScaleCTM(self.fContext, 1., -1.);

   CGContextDrawImage(self.fContext, imageRect, subImage);

   //Restore context state.
   CGContextRestoreGState(self.fContext);

   //imageRep in autorelease pool now.
   CGImageRelease(subImage);
}

//______________________________________________________________________________
- (void) copyPixmap : (QuartzPixmap *) srcPixmap area : (Rectangle_t) area withMask : (QuartzImage *) mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   using ROOT::MacOSX::X11::AdjustCropArea;
 
   //Check parameters.  
   assert(srcPixmap != nil && "copyPixmap:area:withMask:clipOrigin:toPoint:, srcPixmap parameter is nil");
   
   //More difficult case: pixmap already contains reflected image.
   area.fY = ROOT::MacOSX::X11::LocalYROOTToCocoa(srcPixmap, area.fY) - area.fHeight;
   
   if (!AdjustCropArea(srcPixmap, area)) {
      NSLog(@"QuartzView: -copyPixmap:area:withMask:clipOrigin:toPoint, pixmap and copy are no intersection between pixmap rectangle and cropArea");
      return;
   }

   //Check self.
   assert(self.fContext != 0 && "copyPixmap:area:withMask:clipOrigin:toPoint:, self.fContext is null");
   
   CGImageRef imageFromPixmap = [srcPixmap createImageFromPixmap : area];
   assert(imageFromPixmap != nil && "copyPixmap:area:withMask:clipOrigin:toPoint:, createImageFromPixmap failed");

   //Save context state.
   CGContextSaveGState(self.fContext);
   
   if (mask) {
      assert(mask.fImage != nil && "copyPixmap:area:withMask:clipOrigin:toPoint:, mask.fImage is nil");
      assert(CGImageIsMask(mask.fImage) == true && "copyPixmap:area:withMask:clipOrigin:toPoint:, mask.fImage is not a mask");

      const CGRect clipRect = CGRectMake(clipXY.fX, clipXY.fY, mask.fWidth, mask.fHeight);
      CGContextClipToMask(self.fContext, clipRect, mask.fImage);
   }
   
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(self.fContext, imageRect, imageFromPixmap);

   //Restore context state.
   CGContextRestoreGState(self.fContext);
   
   CGImageRelease(imageFromPixmap);
}


//______________________________________________________________________________
- (void) copyImage : (QuartzImage *) srcImage area : (Rectangle_t) area toPoint : (Point_t) dstPoint
{
   using ROOT::MacOSX::X11::AdjustCropArea;

   assert(srcImage != nil && "copyImage:area:toPoint:, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "copyImage:area:toPoint:, srcImage.fImage is nil");
   assert(self.fContext != 0 && "copyImage:area:toPoint:, fContext is null");

   if (!AdjustCropArea(srcImage, area)) {
      NSLog(@"QuartzView: -copyImage:area:toPoint, image and copy area do not intersect");
      return;
   }

   CGImageRef subImage = 0;
   bool needSubImage = false;
   if (area.fX || area.fY || area.fWidth != srcImage.fWidth || area.fHeight != srcImage.fHeight) {
      needSubImage = true;
      subImage = ROOT::MacOSX::X11::CreateSubImage(srcImage, area);
      if (!subImage) {
         NSLog(@"QuartzView: -copyImage:area:toPoint:, subimage creation failed");
         return;
      }
   } else
      subImage = srcImage.fImage;

   CGContextSaveGState(self.fContext);

   CGContextTranslateCTM(self.fContext, 0., self.fHeight); 
   CGContextScaleCTM(self.fContext, 1., -1.);

   dstPoint.fY = ROOT::MacOSX::X11::LocalYCocoaToROOT(self, dstPoint.fY + area.fHeight);
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(self.fContext, imageRect, subImage);

   CGContextRestoreGState(self.fContext);
   
   if (needSubImage)
      CGImageRelease(subImage);
}

//______________________________________________________________________________
- (void) copy : (NSObject<X11Drawable> *) src area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   assert(src != nil && "copy:area:withMask:clipOrigin:toPoint:, src parameter is nil");
   
   if ([src isKindOfClass : [QuartzWindow class]]) {
      //Forget about mask (can I have it???)
      QuartzWindow *qw = (QuartzWindow *)src;
      //Will not work with OpenGL.
      [self copyView : (QuartzView *)qw.fContentView area : area toPoint : dstPoint];
   } else if ([src isKindOfClass : [QuartzView class]]) {
      //Forget about mask (can I have it???)
      [self copyView : (QuartzView *)src area : area toPoint : dstPoint];
   } else if ([src isKindOfClass : [QuartzPixmap class]]) {
      [self copyPixmap : (QuartzPixmap *)src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
   } else if ([src isKindOfClass : [QuartzImage class]]) {
      [self copyImage : (QuartzImage *)src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
   } else {
      assert(0 && "copy:area:withMask:clipOrigin:toPoint:, src is of unknown type");
   }
}

//______________________________________________________________________________
- (unsigned char *) readColorBits : (Rectangle_t) area
{
   //TODO: make the part, reading pixels
   //from NSBitmapImageRep not so lame.

   using ROOT::MacOSX::X11::AdjustCropArea;

   const NSRect visRect = [self visibleRect];
   Rectangle_t srcRect = {};
   srcRect.fX = visRect.origin.x;
   srcRect.fY = visRect.origin.y;
   srcRect.fWidth = visRect.size.width;
   srcRect.fHeight = visRect.size.height;
   
   if (!AdjustCropArea(srcRect, area)) {
      NSLog(@"QuartzView: -readColorBits:, visible rect of view and copy area do not intersect");
      return 0;
   }

   NSBitmapImageRep *imageRep = [self bitmapImageRepForCachingDisplayInRect : visRect];
   if (!imageRep) {
      NSLog(@"QuartzView: -readColorBits: failed");
      return 0;
   }
   
   CGContextRef ctx = self.fContext; //Save old context if any.
   [self cacheDisplayInRect : visRect toBitmapImageRep : imageRep];
   self.fContext = ctx; //Restore old context.
   //
   const unsigned char *srcData = [imageRep bitmapData];
   //We have a source data now. Let's allocate buffer for ROOT's GUI and convert source data.
   unsigned char *data = new unsigned char[area.fWidth * area.fHeight * 4];//bgra?
   const NSInteger bitsPerPixel = [imageRep bitsPerPixel];
   //TODO: ohhh :(((
   assert(bitsPerPixel == 32 && "-readColorBits:, no alpha channel???");

   const NSInteger bytesPerRow = [imageRep bytesPerRow];
   const unsigned dataWidth = bytesPerRow / (bitsPerPixel / 8);//assume an octet :(
   
   unsigned char *dstPixel = data;
   const unsigned char *line = srcData + area.fY * dataWidth * 4;
   const unsigned char *srcPixel = line + area.fX * 4;
      
   for (UShort_t i = 0; i < area.fHeight; ++i) {
      for (UShort_t j = 0; j < area.fWidth; ++j, srcPixel += 4, dstPixel += 4) {
         dstPixel[0] = srcPixel[2];
         dstPixel[1] = srcPixel[1];
         dstPixel[2] = srcPixel[0];
         dstPixel[3] = srcPixel[3];
      }

      line += dataWidth * 4;
      srcPixel = line + area.fX * 4;
   }
   
   return data;
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

//______________________________________________________________________________
- (void) addChild : (NSView<X11Window> *) child
{
   assert(child != nil && "addChild, child view is nil");

   [self addSubview : child];
   child.fParentView = self;
   [child updateLevel : self.fLevel + 1];
}

//______________________________________________________________________________
- (void) getAttributes : (WindowAttributes_t *)attr
{
   assert(attr != 0 && "getAttributes, attr parameter is null");
   
   ROOT::MacOSX::X11::GetWindowAttributes(self, attr);
}

//______________________________________________________________________________
- (void) setAttributes : (const SetWindowAttributes_t *)attr
{
   assert(attr != 0 && "setAttributes, attr parameter is null");

#ifdef DEBUG_ROOT_COCOA
   log_attributes(attr, fID);
#endif

   ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
}

//______________________________________________________________________________
- (void) mapRaised
{
   //Move view to the top of subviews (in UIKit there is a special method).   
   QuartzView *parent = fParentView;
   [self removeFromSuperview];
   [parent addSubview : self];
   [self setHidden : NO];
}

//______________________________________________________________________________
- (void) mapWindow
{   
   [self setHidden : NO];
}

//______________________________________________________________________________
- (void) mapSubwindows
{
   for (QuartzView * v in [self subviews]) {
      [v setHidden : NO]; 
      //[v mapSubwindows];
   }
}

//______________________________________________________________________________
- (void) unmapWindow
{
   [self setHidden : YES];
}

//______________________________________________________________________________
- (BOOL) fIsOverlapped
{
   return fIsOverlapped;
}

//______________________________________________________________________________
- (void) setOverlapped : (BOOL) overlap
{
   fIsOverlapped = overlap;
   for (NSView<X11Window> *child in [self subviews])
      [child setOverlapped : overlap];
}

//______________________________________________________________________________
- (void) raiseWindow
{
   using namespace ROOT::MacOSX::X11;//Comparators.

   for (QuartzView *sibling in [fParentView subviews]) {
      if (self == sibling)
         continue;
      if ([sibling isHidden])
         continue;
      //TODO: equal test is not good :) I have a baaad feeling about this ;)
      if (CGRectEqualToRect(sibling.frame, self.frame)) {
         [sibling setOverlapped : YES];
         //
         [sibling setHidden : YES];
         //
      }
   }

   [self setOverlapped : NO];
   //
   [self setHidden : NO];
   //
   [fParentView sortSubviewsUsingFunction : CompareViewsToRaise context : (void *)self];
   [self setNeedsDisplay : YES];//?
}

//______________________________________________________________________________
- (void) lowerWindow
{
   using namespace ROOT::MacOSX::X11;

   NSEnumerator *reverseEnumerator = [[fParentView subviews] reverseObjectEnumerator];
   for (QuartzView *sibling in reverseEnumerator) {
      if (sibling == self)
         continue;

      //TODO: equal test is not good :) I have a baaad feeling about this ;)
      if (CGRectEqualToRect(sibling.frame, self.frame)) {
         [sibling setOverlapped : NO];
         //
         [sibling setHidden : NO];
         //
         [sibling setNeedsDisplay : YES];
         [self setOverlapped : YES];
         //
         [self setHidden : YES];
         //
         break;
      }
   }
   
   [fParentView sortSubviewsUsingFunction : CompareViewsToLower context : (void*)self];//NO ARC! No __bridge!
}

//______________________________________________________________________________
- (void) updateLevel : (unsigned) newLevel
{
   fLevel = newLevel;
   
   for (QuartzView *child in [self subviews])
      [child updateLevel : fLevel + 1];
}

//______________________________________________________________________________
- (BOOL) isFlipped
{
   //Now view's placement, geometry, moving and resizing can be
   //done with ROOT's (X11) coordinates without conversion.
   return YES;
}

//______________________________________________________________________________
- (void) configureNotifyTree
{
   if (self.fMapState == kIsViewable || fIsOverlapped == YES) {
      if (fEventMask & kStructureNotifyMask) {
         TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
         assert(vx && "configureNotifyTree, gVirtualX is either null or has type different from TGCocoa");
         vx->GetEventTranslator()->GenerateConfigureNotifyEvent(self, self.frame);
      }

      for (NSView<X11Window> *v in [self subviews])
         [v configureNotifyTree];
   }
}

//Key grabs.
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

//Painting mechanics.

//______________________________________________________________________________
- (void) drawRect : (NSRect) dirtyRect
{
   using namespace ROOT::MacOSX::X11;

   (void)dirtyRect;

   if (fID) {
      if (TGWindow *window = gClient->GetWindowById(fID)) {
         if (ViewIsTextViewFrame(self, true))//It's never painted, parent renders child. true == check the parent also.
            return;

         NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
         assert(nsContext != nil && "drawRect, currentContext returned nil");

         TGCocoa *vx = (TGCocoa *)gVirtualX;
         vx->CocoaDrawON();

         fContext = (CGContextRef)[nsContext graphicsPort];
         assert(fContext != 0 && "drawRect, graphicsPort returned null");
         
         CGContextSaveGState(fContext);

         if (window->InheritsFrom("TGContainer"))//It always has an ExposureMask.
            vx->GetEventTranslator()->GenerateExposeEvent(self, [self visibleRect]);

         if (fEventMask & kExposureMask) {
            if (ViewIsTextView(self)) {
               //Send Expose event, using child view (this is how it's done in GUI :( ).
               NSView<X11Window> *viewFrame = FrameForTextView(self);
               if (viewFrame)
                  vx->GetEventTranslator()->GenerateExposeEvent(viewFrame, [viewFrame visibleRect]);//Now we set fExposedRegion for TGView.
            }

            //Ask ROOT's widget/window to draw itself.
            gClient->NeedRedraw(window, kTRUE);
            
            if (!fSnapshotDraw && !ViewIsTextView(self)) {
               //If Cocoa repaints widget, cancel all ROOT's "outside of paint event"
               //rendering into this widget ... Except it's a text view :)
               gClient->CancelRedraw(window);
               vx->GetCommandBuffer()->RemoveGraphicsOperationsForWindow(fID);
            }
         }

         if (fBackBuffer) {
            //Very "special" window.
            CGImageRef image = [fBackBuffer createImageFromPixmap];// CGBitmapContextCreateImage(fBackBuffer.fContext);
            if (image) {
               const CGRect imageRect = CGRectMake(0, 0, fBackBuffer.fWidth, fBackBuffer.fHeight);
               CGContextDrawImage(fContext, imageRect, image);
               CGImageRelease(image);
            }
         }

         CGContextRestoreGState(fContext);         
         vx->CocoaDrawOFF();
#ifdef DEBUG_ROOT_COCOA
         CGContextSetRGBStrokeColor(fContext, 1., 0., 0., 1.);
         CGContextStrokeRect(fContext, dirtyRect);
#endif

         fContext = 0;         
      } else {
#ifdef DEBUG_ROOT_COCOA
         NSLog(@"QuartzView: -drawRect method, no window for id %u was found", fID);
#endif
      }
   }
}

//Event handling.

//______________________________________________________________________________
- (void) setFrame : (NSRect) newFrame
{
   //In case of TBrowser, setFrame started infinite recursion:
   //HandleConfigure for embedded main frame emits signal, slot
   //calls layout, layout calls setFrame -> HandleConfigure and etc. etc.
   if (CGRectEqualToRect(newFrame, self.frame))
      return;

   [super setFrame : newFrame];
}

//______________________________________________________________________________
- (void) setFrameSize : (NSSize) newSize
{
   //Check, if setFrameSize calls setFrame.
   
   [super setFrameSize : newSize];
   
   if ((fEventMask & kStructureNotifyMask) && (self.fMapState == kIsViewable || fIsOverlapped == YES)) {
      TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
      assert(vx != 0 && "setFrameSize:, gVirtualX is either null or has a type, different from TGCocoa");
      vx->GetEventTranslator()->GenerateConfigureNotifyEvent(self, self.frame);
   }

   [self setNeedsDisplay : YES];//?
}

//______________________________________________________________________________
- (void) mouseDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseDown, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "mouseDown, gVirtualX is either null or has a type, different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton1);
}

//______________________________________________________________________________
- (void) scrollWheel : (NSEvent*) theEvent
{
   assert(fID != 0 && "scrollWheel, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "scrollWheel, gVirtualX is either null or has a type, different from TGCocoa");

   const CGFloat deltaY = [theEvent deltaY];
   if (deltaY < 0) {
      vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton5);
      vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton5);
   } else if (deltaY > 0) {
      vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton4);
      vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton4);
   }
}

#ifdef DEBUG_ROOT_COCOA
//______________________________________________________________________________
- (void) printViewInformation
{
   assert(fID != 0 && "printWindowInformation, fID is 0");
   TGWindow *window = gClient->GetWindowById(fID);
   assert(window != 0 && "printWindowInformation, window not found");

   NSLog(@"-----------------View %u info:---------------------", fID);
   NSLog(@"ROOT's window class is %s", window->IsA()->GetName());
   NSLog(@"event mask is:");
   print_mask_info(fEventMask);
   NSLog(@"grab mask is:");
   print_mask_info(fGrabButtonEventMask);
   NSLog(@"----------------End of view info------------------");
}
#endif

//______________________________________________________________________________
- (void) rightMouseDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "rightMouseDown, fID is 0");

#ifdef DEBUG_ROOT_COCOA
   [self printViewInformation];
#endif

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "rightMouseDown, gVirtualX is either null or has type different from TGCocoa");
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
   assert(vx != 0 && "rightMouseUp, gVirtualX is either null or has type different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonReleaseEvent(self, theEvent, kButton2);
}

//______________________________________________________________________________
- (void) mouseEntered : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseEntered, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "mouseEntered, gVirtualX is null or not of TGCocoa type");

   vx->GetEventTranslator()->GenerateCrossingEvent(self, theEvent);  
}

//______________________________________________________________________________
- (void) mouseExited : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseExited, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "mouseExited, gVirtualX is null or not of TGCocoa type");

   vx->GetEventTranslator()->GenerateCrossingEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) mouseMoved : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseMoved, fID is 0");
   
   if (fParentView)//Suppress events in all views, except the top-level one.
      return;      //TODO: check, that it does not create additional problems.

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "mouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) mouseDragged : (NSEvent *)theEvent
{
   assert(fID != 0 && "mouseDragged, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "mouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);   
}

//______________________________________________________________________________
- (void) rightMouseDragged : (NSEvent *)theEvent
{
   assert(fID != 0 && "rightMouseDragged, fID is 0");

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "rightMouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);   
}

//______________________________________________________________________________
- (void) keyDown:(NSEvent *)theEvent
{
   assert(fID != 0 && "keyDown, fID is 0");
  
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != 0 && "keyDown, gVirtualX is null or not of TGCocoa type");
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
   assert(vx != 0 && "becomeFirstResponder, gVirtualX is null or not of TGCocoa type");
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


//Cursors.
//______________________________________________________________________________
- (void) setFCurrentCursor : (ECursor) cursor
{
   if (cursor != fCurrentCursor) {
      fCurrentCursor = cursor;
      [self.fQuartzWindow invalidateCursorRectsForView : self];
   }
}

//______________________________________________________________________________
- (NSCursor *) createCustomCursor 
{
   const char *pngFileName = 0;

   switch (fCurrentCursor) {
   case kMove:
      pngFileName = "move_cursor.png";
      break;
   case kArrowHor:
      pngFileName = "hor_arrow_cursor.png";
      break;
   case kArrowVer:
      pngFileName = "ver_arrow_cursor.png";
      break;
   case kArrowRight:
      pngFileName = "right_arrow_cursor.png";
      break;
   case kBottomLeft:
   case kTopRight:
      pngFileName = "top_right_cursor.png";
      break;
   case kTopLeft:
   case kBottomRight:
      pngFileName = "top_left_cursor.png";
      break;
   default:;
   }
   
   if (pngFileName) {
      const char *path = gSystem->Which("$ROOTSYS/icons", pngFileName, kReadPermission);//This must be deleted.

      if (!path || path[0] == 0) {
         //File was not found.
         delete [] path;
         return nil;
      }
      
      NSString *nsPath = [NSString stringWithFormat : @"%s", path];//in autorelease pool.
      delete [] path;

      NSImage *cursorImage = [[NSImage alloc] initWithContentsOfFile : nsPath];//must call release.

      if (!cursorImage)
         return nil;
      
      NSPoint hotSpot = ROOT::MacOSX::X11::GetCursorHotStop(cursorImage, fCurrentCursor);
      NSCursor *customCursor = [[[NSCursor alloc] initWithImage : cursorImage hotSpot : hotSpot] autorelease];//in autorelease pool.
      
      [cursorImage release];
      
      return customCursor;
   }

   return nil;
}

//______________________________________________________________________________
- (void) resetCursorRects
{
   //Cursors from TVirtaulX:
   // kBottomLeft, kBottomRight, kTopLeft,  kTopRight,
   // kBottomSide, kLeftSide,    kTopSide,  kRightSide,
   // kMove,       kCross,       kArrowHor, kArrowVer,
   // kHand,       kRotate,      kPointer,  kArrowRight,
   // kCaret,      kWatch
   
   NSCursor *cursor = nil;
   
   switch (fCurrentCursor) {
   case kCross:
      cursor = [NSCursor crosshairCursor];
      break;
   case kPointer:
      //Use simple arrow (or this special cursor will be even on GUI widgets).
      break;
   case kHand:
      cursor = [NSCursor openHandCursor];
      break;
   case kLeftSide:
      cursor = [NSCursor resizeLeftCursor];
      break;
   case kRightSide:
      cursor = [NSCursor resizeRightCursor];
      break;
   case kTopSide:
      cursor = [NSCursor resizeUpCursor];
      break;
   case kBottomSide:
      cursor = [NSCursor resizeDownCursor];
      break;
   case kCaret:
      cursor = [NSCursor IBeamCursor];
      break;
   case kRotate:
   case kWatch:
   default:
      cursor = [self createCustomCursor];
   }
   
   if (cursor)
      [self addCursorRect : self.visibleRect cursor : cursor];
   else
      [super resetCursorRects];
}

@end
