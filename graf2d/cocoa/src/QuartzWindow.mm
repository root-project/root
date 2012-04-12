//Author: Timur Pocheptsov 16/02/2012
//#define DEBUG_ROOT_COCOA

#define NDEBUG

#ifdef DEBUG_ROOT_COCOA
#import <iostream>
#import <fstream>

#import "TClass.h"
#endif

#import <stdexcept>
#import <cassert>

#import "QuartzWindow.h"
#import "QuartzPixmap.h"
#import "X11Buffer.h"
#import "X11Events.h"
#import "TGWindow.h"
#import "TGClient.h"
#import "TGCocoa.h"

/*
This class is a stupid work-around to create a snapshot of view's contents.
I do not use QuartzView's drawRect directly, since I had some strange problems
I do not have time to investigate and I've got a 99 problems and a bitchin' one.

Apple's documentation about making a snapshot is useless completely,
you have to try and got all problems before you understand how this crap works.
*/

@interface SnapshotView : NSView
@property (nonatomic, assign) QuartzView *fView;
@end

@implementation SnapshotView
@synthesize fView;

//______________________________________________________________________________
- (void) drawRect : (NSRect) dirtyRect
{
   assert(fView != nil && "drawRect, fView is nil");

   TGWindow *window = gClient->GetWindowById(fView.fID);
   assert(window != nullptr && "drawRect, no window was found");
   
   CGContextRef ctx = (CGContextRef)[[NSGraphicsContext currentContext] graphicsPort];
   assert(ctx != nullptr && "drawRect, bitmap ctx is null");

   fView.fContext = ctx;
   
   TGCocoa *vx = (TGCocoa *)gVirtualX;
   vx->CocoaDrawON();

   CGContextSaveGState(ctx);

   if (window->InheritsFrom("TGContainer"))//It always has an ExposureMask.
      vx->GetEventTranslator()->GenerateExposeEvent(fView, dirtyRect);

   gClient->NeedRedraw(window, kTRUE);

   vx->CocoaDrawOFF();
   CGContextRestoreGState(ctx);
}


@end

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
   assert(attr != nullptr && "GetRootWindowAttributes, attr parameter is null");
   
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
int LocalYCocoaToROOT(QuartzView *parentView, CGFloat yCocoa)
{
   assert(parentView != nil && "LocalYCocoaToROOT, parent view is nil");
   
   return int(parentView.frame.size.height - yCocoa);
}

//______________________________________________________________________________
int LocalYROOTToCocoa(QuartzView *parentView, CGFloat yROOT)
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
NSPoint TranslateToScreen(QuartzView *from, NSPoint point)
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
NSPoint TranslateFromScreen(NSPoint point, QuartzView *to)
{
   assert(to != nil && "TranslateFromScreen, 'to' parameter is nil");
   
   point.y = GlobalYROOTToCocoa(point.y);

   //May be I can use convertBackingTo .... have to check this.
   const NSPoint winPoint = [[to window] convertScreenToBase : point];
   return [to convertPoint : winPoint fromView : nil];
}

//______________________________________________________________________________
NSPoint TranslateCoordinates(QuartzView *from, QuartzView *to, NSPoint sourcePoint)
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
void SetWindowAttributes(const SetWindowAttributes_t *attr, id<X11Drawable> window)
{
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
void GetWindowGeometry(id<X11Drawable> win, WindowAttributes_t *dst)
{
   assert(win != nil && "GetWindowGeometry, win parameter is nil");
   assert(dst != nullptr && "GetWindowGeometry, dst paremeter is null");
   
   dst->fX = win.fX;
   dst->fY = win.fY;
   
   dst->fWidth = win.fWidth;
   dst->fHeight = win.fHeight;
}

//______________________________________________________________________________
void GetWindowAttributes(id<X11Drawable> window, WindowAttributes_t *dst)
{
   assert(window != nil && "GetWindowAttributes, window parameter is nil");
   assert(dst != nullptr && "GetWindowAttributes, attr parameter is null");
   
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

}
}
}

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
}

@synthesize fBackBuffer;


//RootQuartzWindow's life cycle.

//______________________________________________________________________________
- (id) initWithContentRect : (NSRect) contentRect styleMask : (NSUInteger) windowStyle backing : (NSBackingStoreType) bufferingType 
       defer : (BOOL) deferCreation  windowAttributes : (const SetWindowAttributes_t *)attr
{
   self = [super initWithContentRect : contentRect styleMask : windowStyle backing : bufferingType defer : deferCreation];

   if (self) {
      //ROOT's not able to draw GUI concurrently, thanks to global variables and gVirtualX itself.
      [self setAllowsConcurrentViewDrawing : NO];

      //self.delegate = ...
      //create content view here.
      NSRect contentViewRect = contentRect;
      contentViewRect.origin.x = 0.f;
      contentViewRect.origin.y = 0.f;
      fContentView = [[QuartzView alloc] initWithFrame : contentViewRect windowAttributes : 0];
      
      [self setContentView : fContentView];

      [fContentView release];
      
      if (attr)
         ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
   }
   
   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   [super dealloc];
}

///////////////////////////////////////////////////////////
//X11Drawable's protocol.
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
- (QuartzWindow *) fQuartzWindow
{
   return self;
}

//______________________________________________________________________________
- (QuartzView *) fParentView
{
   return nil;
}

//______________________________________________________________________________
- (void) setFParentView : (QuartzView *)parent
{
   (void)parent;
}

//______________________________________________________________________________
- (unsigned) fID 
{
   assert(fContentView != nil && "fID, content view is nil");

   return fContentView.fID;
}

/////////////////////////////////////////////////////////////
//SetWindowAttributes_t/WindowAttributes_t.

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
- (void) setFID : (unsigned) winID
{
   assert(fContentView != nil && "setFID, content view is nil");
   
   fContentView.fID = winID;
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
- (int) fMapState
{
   //Top-level window can be only kIsViewable or kIsUnmapped (not unviewable).
   assert(fContentView != nil && "fMapState, content view is nil");
   
   if ([fContentView isHidden])
      return kIsUnmapped;
      
   return kIsViewable;
}

//______________________________________________________________________________
- (void) copy : (id<X11Drawable>) src area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   assert(fContentView != nil && "copy:area:toPoint:, fContentView is nil");

   [fContentView copy : src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
}


//End of SetWindowAttributes_t/WindowAttributes_t.
/////////////////////////////////////////////////////////////

//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return NO;
}

//______________________________________________________________________________
- (QuartzView *) fContentView
{
   return fContentView;
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
- (NSSize) fSize
{
   //NSWindow's frame includes title-bar.
   //I have to use content view's frame.
   assert(fContentView != nil && "fSize, content view is nil");
   
   return fContentView.frame.size;
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
- (void) addChild : (QuartzView *) child
{
   assert(fContentView != nil && "addChild, content view is nil");
   assert(child != nil && "addChild, child view is nil");
   
   [fContentView addChild : child];
}

//______________________________________________________________________________
- (void) getAttributes : (WindowAttributes_t *) attr
{
   assert(fContentView != nullptr && "getAttributes, content view is nil");
   assert(attr && "getAttributes, attr parameter is nil");

   ROOT::MacOSX::X11::GetWindowAttributes(self, attr);
}

//______________________________________________________________________________
- (void) setAttributes : (const SetWindowAttributes_t *)attr
{
   assert(attr != nullptr && "setAttributes, attr parameter is null");

#ifdef DEBUG_ROOT_COCOA
   log_attributes(attr, self.fID);
#endif

   ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
}

//______________________________________________________________________________
- (void) mapRaised
{
   assert(fContentView && "mapRaised, content view is nil");

   [self orderFront : self];
   [fContentView setHidden : NO];
   [fContentView configureNotifyTree];
}

//______________________________________________________________________________
- (void) mapWindow
{
   assert(fContentView != nil && "mapWindow, content view is nil");

   [self orderFront : self];
   [fContentView setHidden : NO];
   [fContentView configureNotifyTree];
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

//End of X11Drawable's protocol.
///////////////////////////////////////////////////////////


//NSWindowDelegate's methods here.

@end

//
//
//
//

@implementation QuartzView

@synthesize fBackBuffer;
@synthesize fParentView;
@synthesize fLevel;
@synthesize fIsOverlapped;
@synthesize fID;

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

@synthesize fGrabButton;
@synthesize fGrabButtonEventMask;
@synthesize fGrabKeyModifiers;
@synthesize fOwnerEvents;

@synthesize fContext;

//______________________________________________________________________________
- (id) initWithFrame : (NSRect) frame windowAttributes : (const SetWindowAttributes_t *)attr
{
   if (self = [super initWithFrame : frame]) {
      //Make this explicit (though memory is zero initialized).
      fID = 0;
      fLevel = 0;
      
      //Passive grab parameters.
      fGrabButton = -1;//0 is kAnyButton.
      fGrabButtonEventMask = 0;
      fOwnerEvents = NO;
      
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
   }
   
   return self;
}

//______________________________________________________________________________
- (QuartzWindow *) fQuartzWindow
{
   return (QuartzWindow *)[self window];
}

//______________________________________________________________________________
- (BOOL) isFlipped
{
   //Now view's placement, geometry, moving and resizing can be
   //done with ROOT's (X11) coordinates without conversion.
   return YES;
}


/////////////////////////////////////////////////////////////
//X11Drawable protocol.

//______________________________________________________________________________
- (BOOL) fIsPixmap
{
   return NO;
}

//______________________________________________________________________________
- (QuartzView *) fContentView
{
   return self;
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
- (NSSize) fSize
{
   return self.frame.size;
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
- (void) addChild : (QuartzView *) child
{
   assert(child != nil && "addChild, child view is nil");

   [self addSubview : child];
   child.fParentView = self;
   [child updateLevel : self.fLevel + 1];
}

//______________________________________________________________________________
- (void) getAttributes : (WindowAttributes_t *)attr
{
   assert(attr != nullptr && "getAttributes, attr parameter is null");
   
   ROOT::MacOSX::X11::GetWindowAttributes(self, attr);
}

//______________________________________________________________________________
- (void) setAttributes : (const SetWindowAttributes_t *)attr
{
   assert(attr != nullptr && "setAttributes, attr parameter is null");

#ifdef DEBUG_ROOT_COCOA
   log_attributes(attr, fID);
#endif

   ROOT::MacOSX::X11::SetWindowAttributes(attr, self);
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
      [v mapSubwindows];
   }
}

//______________________________________________________________________________
- (void) unmapWindow
{
   [self setHidden : YES];
}

//______________________________________________________________________________
- (void) setOverlapped : (BOOL) overlap
{
   fIsOverlapped = overlap;
   for (QuartzView *child in [self subviews])
      [child setOverlapped : overlap];
}

//______________________________________________________________________________
- (void) raiseWindow
{
   using namespace ROOT::MacOSX::X11;//Comparators.

   for (QuartzView *sibling in [fParentView subviews]) {
      if (self == sibling)
         continue;
      //TODO: equal test is not good :) I have a baaad feeling about this ;)
      if (CGRectEqualToRect(sibling.frame, self.frame))
         [sibling setOverlapped : YES];
   }

   [self setOverlapped : NO];
   [fParentView sortSubviewsUsingFunction : CompareViewsToRaise context : (void *)self];//ARC will complain, but ... NO ARC!!! :)))
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
         [sibling setNeedsDisplay : YES];
         [self setOverlapped : YES];
         break;
      }
   }
   
   [fParentView sortSubviewsUsingFunction : CompareViewsToLower context : (void*)self];//NO ARC! No __bridge!
}

//______________________________________________________________________________
- (void) configureNotifyTree
{
   if (self.fMapState == kIsViewable) {
      if (fEventMask & kStructureNotifyMask) {
         TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
         assert(vx && "configureNotifyTree, gVirtualX is either null or has type different from TGCocoa");
         vx->GetEventTranslator()->GenerateConfigureNotifyEvent(self, self.frame);
      }

      for (QuartzView * v in [self subviews])
         [v configureNotifyTree];
   }
}

//______________________________________________________________________________
- (void) updateLevel : (unsigned) newLevel
{
   fLevel = newLevel;
   
   for (QuartzView *child in [self subviews])
      [child updateLevel : fLevel + 1];
}

//______________________________________________________________________________
- (void) copyImage : (QuartzImage *) srcImage area : (Rectangle_t) area withMask : (QuartzImage *) mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   //Check parameters.
   assert(srcImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage parameter is nil");
   assert(srcImage.fImage != nil && "copyImage:area:withMask:clipOrigin:toPoint:, srcImage.fImage is nil");

   //Check self.
   assert(self.fContext != nullptr && "copyImage:area:withMask:clipOrigin:toPoint:, self.fContext is null");
   
   CGImageRef subImage = nullptr;
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

   const NSRect visRect = [srcView visibleRect];
   SnapshotView *snapshot = [[SnapshotView alloc] initWithFrame : visRect];
   NSBitmapImageRep *imageRep = [snapshot bitmapImageRepForCachingDisplayInRect : visRect];
   if (!imageRep) {
      NSLog(@"QuartzView: -copyView:area:toPoint failed");
      return;
   }
   
   assert(srcView != nil && "copyView:area:toPoint:, srcView parameter is nil");
   assert(self.fContext != nullptr && "copyView:area:toPoint, self.fContext is null");

   //It can happen, that src and self are the same.
   //cacheDisplayInRect calls drawRect with bitmap context 
   //(and this will reset self.fContext: I have to save/restore it.
   CGContextRef ctx = srcView.fContext;

   snapshot.fView = srcView;
   [snapshot cacheDisplayInRect : visRect toBitmapImageRep : imageRep];
   srcView.fContext = ctx;

   const CGRect subImageRect = CGRectMake(area.fX, area.fY, area.fWidth, area.fHeight);
   CGImageRef subImage = CGImageCreateWithImageInRect(imageRep.CGImage, subImageRect);

   CGContextSaveGState(self.fContext);
   const CGRect imageRect = CGRectMake(dstPoint.fX, dstPoint.fY, area.fWidth, area.fHeight);
   CGContextDrawImage(self.fContext, imageRect, subImage);

   CGContextFlush(self.fContext);
   
   [snapshot release];

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
   assert(self.fContext != nullptr && "copyPixmap:area:withMask:clipOrigin:toPoint:, self.fContext is null");
   
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
   assert(self.fContext != nullptr && "copyImage:area:toPoint:, fContext is null");

   if (!AdjustCropArea(srcImage, area)) {
      NSLog(@"QuartzView: -copyImage:area:toPoint, image and copy area do not intersect");
      return;
   }

   CGImageRef subImage = nullptr;
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
- (void) copy : (id<X11Drawable>) src area : (Rectangle_t) area withMask : (QuartzImage *)mask clipOrigin : (Point_t) clipXY toPoint : (Point_t) dstPoint
{
   assert(src != nil && "copy:area:withMask:clipOrigin:toPoint:, src parameter is nil");
   
   NSObject<X11Drawable> *srcObj = (NSObject<X11Drawable> *)src;

   if ([srcObj isKindOfClass : [QuartzWindow class]]) {
      //Forget about mask (can I have it???)
      [self copyView : srcObj.fContentView area : area toPoint : dstPoint];
   } else if ([srcObj isKindOfClass : [QuartzView class]]) {
      //Forget about mask (can I have it???)
      [self copyView : (QuartzView *)srcObj area : area toPoint : dstPoint];
   } else if ([srcObj isKindOfClass : [QuartzPixmap class]]) {
      [self copyPixmap : (QuartzPixmap *)src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
   } else if ([srcObj isKindOfClass : [QuartzImage class]]) {
      [self copyImage : (QuartzImage *)src area : area withMask : mask clipOrigin : clipXY toPoint : dstPoint];
   } else {
      assert(0 && "copy:area:withMask:clipOrigin:toPoint:, src is of unknown type");
   }
}

//End of X11Drawable protocol.
/////////////////////////////////////////////////////////////

//Painting mechanics.

//______________________________________________________________________________
- (void) drawRect : (NSRect) dirtyRect
{
   (void)dirtyRect;

   if (fID) {
      if (TGWindow *window = gClient->GetWindowById(fID)) {
         NSGraphicsContext *nsContext = [NSGraphicsContext currentContext];
         assert(nsContext != nil && "drawRect, currentContext returned nil");

         TGCocoa *vx = (TGCocoa *)gVirtualX;
         vx->CocoaDrawON();

         fContext = (CGContextRef)[nsContext graphicsPort];
         assert(fContext != nullptr && "drawRect, graphicsPort returned null");
         
         CGContextSaveGState(fContext);

         if (window->InheritsFrom("TGContainer"))//It always has an ExposureMask.
            vx->GetEventTranslator()->GenerateExposeEvent(self, [self visibleRect]);

         if (fEventMask & kExposureMask) {
            //Ask ROOT's widget/window to draw itself.
            gClient->NeedRedraw(window, kTRUE);
            gClient->CancelRedraw(window);
            vx->GetCommandBuffer()->RemoveGraphicsOperationsForWindow(fID);
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
         fContext = nullptr;
      } else {
         NSLog(@"QuartzView: -drawRect method, no window for id %u was found", fID);
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
   
   if ((fEventMask & kStructureNotifyMask) && self.fMapState == kIsViewable) {
      TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
      assert(vx != nullptr && "setFrameSize:, gVirtualX is either null or has type different from TGCocoa");
      vx->GetEventTranslator()->GenerateConfigureNotifyEvent(self, self.frame);
   }

   [self setNeedsDisplay : YES];//?
}

//______________________________________________________________________________
- (void) mouseDown : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseDown, fID is 0");
   
   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseDown, gVirtualX is either null or has type different from TGCocoa");
   vx->GetEventTranslator()->GenerateButtonPressEvent(self, theEvent, kButton1);
}

#ifdef DEBUG_ROOT_COCOA
//______________________________________________________________________________
- (void) printViewInformation
{
   assert(fID != 0 && "printWindowInformation, fID is 0");
   TGWindow *window = gClient->GetWindowById(fID);
   assert(window != nullptr && "printWindowInformation, window not found");

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
   
   (void)theEvent;//TODO: delete.

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
- (BOOL) acceptsFirstResponder
{
   //Temporary version, will be more complex (grabs, focus, etc.).
   return YES;
}

//______________________________________________________________________________
- (void) mouseMoved : (NSEvent *) theEvent
{
   assert(fID != 0 && "mouseMoved, fID is 0");
   
   if (fParentView)//Suppress events in all views, except the top-level one.
      return;      //TODO: check, that it does not create additional problems.

   TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
   assert(vx != nullptr && "mouseMoved, gVirtualX is null or not of TGCocoa type");
   
   vx->GetEventTranslator()->GeneratePointerMotionEvent(self, theEvent);
}

//______________________________________________________________________________
- (void) mouseDragged : (NSEvent *)theEvent
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
- (void) rightMouseDragged : (NSEvent *)theEvent
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
- (void) keyDown:(NSEvent *)theEvent
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

@end
