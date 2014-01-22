#include <cassert>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include <string>
#include <list>

#include <sys/time.h>

#include <Cocoa/Cocoa.h>

#include "CocoaUtils.h"

//ROOTSplashScreenView: content view for our panel (splash screen window)
//with a background (ROOT's logo) + scrollview and textview to show info
//about contributors.

@interface ROOTSplashScreenView : NSView
@end

@implementation ROOTSplashScreenView {
   NSImage *backgroundImage;
   NSScrollView *scrollView;
   NSTextView *textView;
}

//_________________________________________________________________
- (id) initWithImage : (NSImage *) image
{
   assert(image != nil && "initWithImage:, parameter 'image' is nil");
   
   const CGSize imageSize = image.size;
   
   //minimal sizes required by text view's position (which is 'hardcoded' and
   //must be the same as in rootxx (X11 version).
   assert(imageSize.width >= 300 && imageSize.height >= 285 &&
          "initWithImage:, unexpected background image sizes");

   self = [super initWithFrame : CGRectMake(0, 0, imageSize.width, imageSize.height)];
   
   if (self) {
      //Let's create our child views.
      backgroundImage = [image retain];

      //In rootxx it's: x == 15, y == 155, w == 285, h == 130 (top-left corner).
      //In Cocoa it's counted from the bottom left: y = height - yX11 - h
      CGRect scrollRect = CGRectMake(15., imageSize.height - 285., 285., 130.);
      scrollView = [[NSScrollView alloc] initWithFrame : scrollRect];
      [self addSubview : scrollView];
      
      scrollRect.origin = CGPoint();
      textView = [[NSTextView alloc] initWithFrame : scrollRect];
      [scrollView addSubview : textView];
   }
   
   return self;
}

//_________________________________________________________________
- (void) dealloc
{
   [backgroundImage release];
   [textView release];
   [scrollView release];
   
   [super dealloc];
}

//_________________________________________________________________
- (void) drawRect : (NSRect) rect
{
#pragma unused(rect)
   assert(backgroundImage != nil && "drawRect:, backgroundImage is nil");
   
   CGRect frame = self.frame;
   frame.origin = CGPoint();

   const CGSize imageSize = backgroundImage.size;
   [backgroundImage drawInRect : frame
                    fromRect : CGRectMake(0., 0., imageSize.width, imageSize.height)
                    operation : NSCompositeSourceOver
                    fraction : 1.];
}

@end

//To be set from a signal handler.
namespace {

volatile sig_atomic_t popdown = 0;

}


//Our "top-level-window" - borderless panel.

@interface ROOTSplashScreenPanel : NSPanel
@end

@implementation ROOTSplashScreenPanel

#pragma mark - Events.

//_________________________________________________________________
- (BOOL) canBecomeMainWindow
{
   return YES;
}

//_________________________________________________________________
- (BOOL) canBecomeKeyWindow
{
   return YES;
}

//_________________________________________________________________
- (void) sendEvent : (NSEvent *) theEvent
{
   if ([theEvent type] == NSKeyDown) {
      popdown = 1;
      //TODO: something else actually, not popdown.
      return;
   }

   if ([theEvent type] == NSLeftMouseDown || [theEvent type] == NSRightMouseDown) {
      popdown = 1;
      //TODO: something else, not popdown.
      return;
   }

   [super sendEvent : theEvent];
}

@end

namespace {

bool showAboutInfo = false;

ROOTSplashScreenPanel *splashScreen = nil;
//Top-level autorelease pool.
NSAutoreleasePool * topLevelPool = nil;

//We use a signal timer to check:
//if we got a SIGUSR1 (see rootx.cxx) or
//if we have to remove a splash-screen after a delay.
CFRunLoopTimerRef signalTimer = 0;
const CFTimeInterval signalInterval = 1.;

timeval popupCreationTime;
const int splashScreenDelay = 4000;//4 seconds as in rootx.cxx.

//Timer for a scroll animation.
CFRunLoopTimerRef scrollTimer = 0;
const CFTimeInterval scrollInterval = 2.;

//Timer callbacks 'fire' custom NSEvents:
enum TimerEventType {//make it enum class when C++11 is here.
   kScrollTimer = 1,
   kSignalTimer = 2,
   kRemoveSplashTimer = 3
};

//Aux. functions:
bool InitCocoa();
bool InitTimers();
bool StayUp();
void ProcessTimerEvent(NSEvent *event);
bool ReadContributors(std::list<std::string> & contributors);

}//unnamed namespace.


//Platform-specific (OS X) versions of PopupLogo, WaitLogo, PopdownLogo, CloseDisplay.
//_________________________________________________________________
void PopupLogo(bool about)
{
#pragma unused(about)

   //Let's do something stupid to suppress 'unused' warnings.
   (void)showAboutInfo;
   if (false) {
      InitCocoa();
      InitTimers();
      ProcessTimerEvent(nil);
      std::list<std::string> lst;
      ReadContributors(lst);
   }
   //
   
   //Noop at the moment.
   
   
   /*
   if (!InitCocoa()) {
      //TODO: diagnostic.
      return;
   }
   
   std::list<std::string> contributors;
   if (!ReadContributors(contributors)) {
      //TODO: diagnostic.
      return;
   }
   
   if (gettimeofday(&popupCreationTime, 0) == -1) {
      //TODO: check errno and issue a message,
      //we need a valid popup creation time.
      return;
   }

   //Image first.
#ifdef ROOTICONPATH
   const std::string fileName(std::string(ROOTICONPATH) + "/Splash.gif");
#else
   const char * const env = std::getenv("ROOTSYS");
   if (!env) {
      //TODO: diagnostic.
      return;
   }
   
   const std::string fileName(std::string(env) + "/icons/Splash.gif");
#endif

   using ROOT::MacOSX::Util::NSScopeGuard;

   const NSScopeGuard<NSString> nsStringGuard([[NSString alloc] initWithFormat : @"%s", fileName.c_str()]);
   if (!nsStringGuard.Get()) {
      //TODO: diagnostic.
      return;
   }
   
   const NSScopeGuard<NSImage> imageGuard([[NSImage alloc] initWithContentsOfFile : nsStringGuard.Get()]);
   if (!imageGuard.Get()) {
      //TODO: diagnostic.
      return;
   }
   
   const CGSize imageSize = imageGuard.Get().size;
   //These sizes are from X11 version, they are related to the geometry of a scroll view.
   if (imageSize.width < 300 || imageSize.height < 285) {
      //TODO: diagnostic.
      return;
   }
   
   //Create a panel.
   NSScopeGuard<ROOTSplashScreenPanel> splashGuard([[ROOTSplashScreenPanel alloc]
                                                    initWithContentRect : CGRectMake(0, 0, imageSize.width, imageSize.height)
                                                    styleMask : NSBorderlessWindowMask
                                                    backing : NSBackingStoreBuffered
                                                    defer : NO]);
   if (!splashGuard.Get()) {
      //TODO: diagnostic.
      return;
   }
   
   const NSScopeGuard<ROOTSplashScreenView> viewGuard([[ROOTSplashScreenView alloc] initWithImage : imageGuard.Get()]);
   if (!viewGuard.Get()) {
      //TODO: diagnostic.
      return;
   }
   
   [splashGuard.Get() setContentView : viewGuard.Get()];
   splashScreen = splashGuard.Get();
   splashGuard.Release();

   //Now, set the text in a text view.
   if (NSScreen * const screen = [NSScreen mainScreen]) {
      const CGSize screenSize = screen.frame.size;
      const NSRect frame = splashScreen.frame;
      const CGPoint origin = CGPointMake(screenSize.width / 2 - frame.size.width / 2, screenSize.height / 2 - frame.size.height / 2);
      [splashScreen setFrameOrigin : origin];
   }//else - is it possible?
   
   [splashScreen makeKeyAndOrderFront : nil];*/
}

//_________________________________________________________________
void PopdownLogo()
{
   //This function is called from the signal handler.

   //Noop.
}

//_________________________________________________________________
void WaitLogo()
{
   //Noop at the moment.
   if (!splashScreen)
      //TODO: diagnostic.
      return;
   
   /*
   if (!InitTimers())
      //TODO: diagnostic.
      return;
   */
   
   //1. Attach timers to the main run loop.
   //2. Event loop.
   //3. Cleanup.
}

//_________________________________________________________________
void CloseDisplay()
{
   //Noop.
}

//Aux. functions.

extern "C" {

//_________________________________________________________________
void ROOTSplashscreenTimerCallback(CFRunLoopTimerRef timer, void *info)
{
#pragma unused(info)
   if (timer == signalTimer) {
      //
      if (popdown) {
         //Check if signal handler set popdown flag.
         //We have to stop everything and inform our main loop.
         NSEvent * const timerEvent = [NSEvent otherEventWithType : NSApplicationDefined location : NSMakePoint(0, 0) modifierFlags : 0
                                       timestamp: 0. windowNumber : 0 context : nil subtype : 0 data1 : kSignalTimer data2 : 0];
         [NSApp postEvent : timerEvent atStart : NO];
      } else if (!StayUp()) {
         //Check if we have to hide a splash screen after delay.
         //Oook, we have to stop everything and inform our main loop.

         NSEvent * const timerEvent = [NSEvent otherEventWithType : NSApplicationDefined location : NSMakePoint(0, 0) modifierFlags : 0
                                       timestamp: 0. windowNumber : 0 context : nil subtype : 0 data1 : kRemoveSplashTimer data2 : 0];
         [NSApp postEvent : timerEvent atStart : NO];
      }
   }
}

}//extern "C"

namespace {

//_________________________________________________________________
bool InitCocoa()
{
   if (!topLevelPool) {
      //
      //It's not clear, if I need TransformProcessType at all or activateIgnoring is enough.
      //Anyway, let's try.
      ProcessSerialNumber psn = {0, kCurrentProcess};
      const OSStatus res = ::TransformProcessType(&psn, kProcessTransformToUIElementApplication);
      if (res != noErr && res != paramErr) {
         //TODO: diagnostic.
         return false;
      }

      [[NSApplication sharedApplication] setActivationPolicy : NSApplicationActivationPolicyAccessory];
      [[NSApplication sharedApplication] activateIgnoringOtherApps : YES];

      topLevelPool = [[NSAutoreleasePool alloc] init];
   }
   
   return true;
}

//_________________________________________________________________
bool InitTimers()
{
   assert(scrollTimer == 0 && "InitTimers, scrollTimer was initialized already");
   assert(signalTimer == 0 && "InitTimers, signalTimer was initialized already");

   using ROOT::MacOSX::Util::CFScopeGuard;
   
   CFScopeGuard<CFRunLoopTimerRef> guard1(CFRunLoopTimerCreate(kCFAllocatorDefault,//allocator
                                                               CFAbsoluteTimeGetCurrent() + signalInterval,//fireDate
                                                               signalInterval,//interval in seconds(?)
                                                               0,//flags - not used
                                                               0,//order - not used
                                                               ROOTSplashscreenTimerCallback,
                                                               0//info
                                                               ));
   if (!guard1.Get())
      return false;
   
   CFScopeGuard<CFRunLoopTimerRef> guard2(CFRunLoopTimerCreate(kCFAllocatorDefault,
                                                               CFAbsoluteTimeGetCurrent() + scrollInterval,
                                                               scrollInterval,
                                                               0,
                                                               0,
                                                               ROOTSplashscreenTimerCallback,
                                                               0
                                                               ));

   if (!guard2.Get())
      return false;
   
   //TODO: Hmm, may be, it's not a bad idea to return a pointer from 'Release'?
   scrollTimer = guard1.Get();
   guard1.Release();
   signalTimer = guard2.Get();
   guard2.Release();
   
   return true;
}

//_________________________________________________________________
bool ReadContributors(std::list<std::string> & contributors)
{
#ifdef ROOTDOCDIR
   const std::string fileName(std::string(ROOTDOCDIR) + "/CREDITS");
#else
   const char * const env = std::getenv("ROOTSYS");
   if (!env)
      //TODO: diagnostic?
      return false;
   
   const std::string fileName(std::string(env) + "/README/CREDITS");
#endif

   std::ifstream inputFile(fileName.c_str());
   if (!inputFile)
      return false;

   std::list<std::string> tmp;
   std::string line(200, ' ');
   while (std::getline(inputFile, line)) {
      if (line.length() > 3) {
         if (line[0] == 'N' && line[1] == ':' && line[2] == ' ')
            tmp.push_back(line);
      }
   }
   
   tmp.swap(contributors);
   
   return true;
}

//_________________________________________________________________
void ProcessTimerEvent(NSEvent *event)
{
//   assert(event != nil && "ProcessTimerEvent, parameter 'event' is nil");
#pragma unused(event)
   //TODO: scroll event.
}

//_________________________________________________________________
bool StayUp()
{
   //Taken from rootxx.cxx.

   timeval ctv = {};
   timeval dtv = {};
   timeval tv = {};
   timeval ptv = popupCreationTime;

   tv.tv_sec  = splashScreenDelay / 1000;
   tv.tv_usec = (splashScreenDelay % 1000) * 1000;

   gettimeofday(&ctv, 0);
   if ((dtv.tv_usec = ctv.tv_usec - ptv.tv_usec) < 0) {
      dtv.tv_usec += 1000000;
      ptv.tv_sec++;
   }

   dtv.tv_sec = ctv.tv_sec - ptv.tv_sec;
   if ((ctv.tv_usec = tv.tv_usec - dtv.tv_usec) < 0) {
      ctv.tv_usec += 1000000;
      dtv.tv_sec++;
   }
   
   ctv.tv_sec = tv.tv_sec - dtv.tv_sec;

   if (ctv.tv_sec < 0)
      return false;

   return true;
}

}//unnamed namespace.
