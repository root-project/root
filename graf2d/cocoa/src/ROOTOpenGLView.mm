#import "ROOTOpenGLView.h"
#import "QuartzWindow.h"
#import "X11Events.h"
#import "TGCocoa.h"

@implementation ROOTOpenGLView {
   NSMutableArray *fPassiveKeyGrabs;
   BOOL            fIsOverlapped;
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

//______________________________________________________________________________
- (id) initWithFrame : (NSRect) frameRect pixelFormat : (NSOpenGLPixelFormat *) format
{
   (void)format;

   if (self = [super initWithFrame : frameRect]) {
      fPassiveKeyGrabs = [[NSMutableArray alloc] init];
      [self setHidden : YES];//Not sure.
      fCurrentCursor = kPointer;
      //Tracking area?
   }
   
   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   [fPassiveKeyGrabs release];
   [super dealloc];
}

//______________________________________________________________________________
- (void) clearGLContext
{
}

//______________________________________________________________________________
- (NSOpenGLContext *) openGLContext
{
   return nil;
}

//______________________________________________________________________________
- (void) setOpenGLContext : (NSOpenGLContext *) context
{
   (void)context;
}

//______________________________________________________________________________
- (NSOpenGLPixelFormat *) pixelFormat
{
   return nil;
}

//______________________________________________________________________________
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat
{
   (void)pixelFormat;
}

//______________________________________________________________________________
- (void) update
{
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

////////
//Shared methods:

#import "SharedViewMethods.h"

@end

