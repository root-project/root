#import <cassert>

#import <OpenGL/gl.h>

#import "ROOTOpenGLView.h"
#import "QuartzWindow.h"
#import "X11Events.h"
#import "TGCocoa.h"

@implementation ROOTOpenGLView {
   NSMutableArray *fPassiveKeyGrabs;
   BOOL            fIsOverlapped;
   
   NSOpenGLPixelFormat *fPixelFormat;
   NSOpenGLContext *fOpenGLContext;
   BOOL fCtxIsCurrent;
}

@synthesize fID;
@synthesize fEventMask;
@synthesize fParentView;
@synthesize fLevel;
@synthesize fGrabButton;
@synthesize fGrabButtonEventMask;
@synthesize fGrabKeyModifiers;
@synthesize fOwnerEvents;
@synthesize fCurrentCursor;
@synthesize fDepth;
@synthesize fBitGravity;
@synthesize fWinGravity;
@synthesize fClass;

//______________________________________________________________________________
- (id) initWithFrame : (NSRect) frameRect pixelFormat : (NSOpenGLPixelFormat *) format
{
   if (self = [super initWithFrame : frameRect]) {
      fPassiveKeyGrabs = [[NSMutableArray alloc] init];
      [self setHidden : YES];//Not sure.
      fCurrentCursor = kPointer;
      fIsOverlapped = NO;
      fPixelFormat = [format retain];
      fCtxIsCurrent = kFALSE;
      //Tracking area?
   }

   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   [fPassiveKeyGrabs release];
   [fPixelFormat release];
   [fOpenGLContext release];

   [super dealloc];
}

//______________________________________________________________________________
- (void) clearGLContext
{
   //[NSOpenGLContext clearCurrentContext];
}

//______________________________________________________________________________
- (NSOpenGLContext *) openGLContext
{
   return fOpenGLContext;
}

//______________________________________________________________________________
- (void) setOpenGLContext : (NSOpenGLContext *) context
{
   if (context != fOpenGLContext) {
      [fOpenGLContext release];
      //
      fOpenGLContext = [context retain];
      if (![self isHidden])
         [fOpenGLContext setView : self];
   }
}

//______________________________________________________________________________
- (void) makeContextCurrent
{
   fCtxIsCurrent = NO;
   if (!fOpenGLContext)
      return;
   
   if ([fOpenGLContext view] != self)
      [fOpenGLContext setView : self];
   
   [fOpenGLContext makeCurrentContext];
   fCtxIsCurrent = YES; 
}

//______________________________________________________________________________
- (void) flushGLBuffer 
{
   assert(fOpenGLContext == [NSOpenGLContext currentContext] && "flushGLBuffer, view's GL context is not current");
   //
   glFlush();//???
   [fOpenGLContext flushBuffer];
}

//______________________________________________________________________________
- (NSOpenGLPixelFormat *) pixelFormat
{
   return fPixelFormat;
}

//______________________________________________________________________________
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat
{
   (void)pixelFormat;
   //Do not modify fPixelFormat.
}

//______________________________________________________________________________
- (void) update
{
}

//______________________________________________________________________________
- (BOOL) isGLContextCurrent
{
   return fCtxIsCurrent;
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
   return YES;
}

//______________________________________________________________________________
- (void) getAttributes : (WindowAttributes_t *) attr
{
   assert(attr && "getAttributes, attr parameter is nil");
   ROOT::MacOSX::X11::GetWindowAttributes(self, attr);
}

//______________________________________________________________________________
- (void) mapWindow
{   
   [self setHidden : NO];
}

//______________________________________________________________________________
- (void) mapSubwindows
{
   //GL-view can not have any subwindows.
   assert([[self subviews] count] == 0 && "mapSubwindows, GL-view has children");
}

//______________________________________________________________________________
- (void) configureNotifyTree
{
   //The only node in the tree is 'self'.
   if (self.fMapState == kIsViewable) {
      if (fEventMask & kStructureNotifyMask) {
         TGCocoa *vx = dynamic_cast<TGCocoa *>(gVirtualX);
         assert(vx && "configureNotifyTree, gVirtualX is either null or has type different from TGCocoa");
         vx->GetEventTranslator()->GenerateConfigureNotifyEvent(self, self.frame);
      }
   }
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
   [self setHidden : fIsOverlapped];
}

//______________________________________________________________________________
- (void) updateLevel : (unsigned) newLevel
{
   fLevel = newLevel;
}

////////
//Shared methods:

#import "SharedViewMethods.h"

@end

