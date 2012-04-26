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

   if (self = [self initWithFrame : frameRect pixelFormat : format]) {
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

//NSOpenGLView's overriders.

//______________________________________________________________________________
- (void) clearGLContext
{
   [super clearGLContext];
}

//______________________________________________________________________________
- (NSOpenGLContext *) openGLContext
{
   return [super openGLContext];
}

//______________________________________________________________________________
- (NSOpenGLPixelFormat *) pixelFormat
{
   return [super pixelFormat];
}

//______________________________________________________________________________
- (void) prepareOpenGL
{
   [super prepareOpenGL];
}

//______________________________________________________________________________
- (void) reshape
{
   [super reshape];
}

//______________________________________________________________________________
- (void) setOpenGLContext : (NSOpenGLContext *) context
{
   [super setOpenGLContext : context];
}

//______________________________________________________________________________
- (void) setPixelFormat : (NSOpenGLPixelFormat *) pixelFormat
{
   [super setPixelFormat : pixelFormat];
}

//______________________________________________________________________________
- (void) update
{
   [super update];
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

