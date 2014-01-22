#include <cassert>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include <string>
#include <cerrno>
#include <list>

#include <sys/types.h>
#include <sys/wait.h>
#include <sys/time.h>

#include <Cocoa/Cocoa.h>

#include "CocoaUtils.h"

//ROOTSplashScreenView: content view for our panel (splash screen window)
//with a background (ROOT's logo) + scrollview and textview to show info
//about contributors.

/*
namespace ROOT {
namespace ROOTX {

//This had internal linkage before, now must be accessible from rootx-cocoa.mm.
extern int gChildpid;

}
}
*/

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

bool popupDone = false;

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
      popupDone = true;
      return;
   }

   if ([theEvent type] == NSLeftMouseDown || [theEvent type] == NSRightMouseDown) {
      popupDone = true;
      return;
   }

   [super sendEvent : theEvent];
}

@end

namespace {

volatile sig_atomic_t popdown = 0;
bool showAboutInfo = false;

ROOTSplashScreenPanel *splashScreen = nil;
//Top-level autorelease pool.
NSAutoreleasePool * topLevelPool = nil;

//We use a signal timer to check:
//if we got a SIGUSR1 (see rootx.cxx) or
//if we have to remove a splash-screen after a delay.
CFRunLoopTimerRef signalTimer = 0;
const CFTimeInterval signalInterval = 0.1;

timeval popupCreationTime;
const CFTimeInterval splashScreenDelayInSec = 4.;//4 seconds as in rootx.cxx.

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
bool InitTimers(bool background);
void RunEventLoop();
void RunEventLoopInBackground();
bool StayUp();
bool CreateSplashscreen();
void SetSplashscreenPosition();
//Non GUI aux. function.
bool ReadContributors(std::list<std::string> & contributors);

}//unnamed namespace.


//Platform-specific (OS X) versions of PopupLogo, WaitLogo, PopdownLogo, CloseDisplay.
//_________________________________________________________________
void PopupLogo(bool about)
{
#pragma unused(about)
return;//Still Noop!!!

   if (!InitCocoa()) {
      //TODO: diagnostic.
      return;
   }
   
   std::list<std::string> contributors;
   if (!ReadContributors(contributors)) {
      //TODO: diagnostic.
      return;
   }
   
   //0. For StayUp to check when we should hide our splash-screen.
   if (gettimeofday(&popupCreationTime, 0) == -1) {
      //TODO: check errno and issue a message,
      //we need a valid popup creation time.
      return;
   }

   if (!CreateSplashscreen()) {
      //TODO: diagnostic.
      return;
   }
   
   SetSplashscreenPosition();

   [splashScreen makeKeyAndOrderFront : nil];

   //
   showAboutInfo = about;
}

//_________________________________________________________________
void PopdownLogo()
{
   //This function is called from the signal handler.
   popdown = 1;
}

//_________________________________________________________________
void WaitLogo()
{
return;//Still Noop!!!

   if (!splashScreen)
      //TODO: diagnostic.
      return;

   RunEventLoop();
   
   //Cleanup.
   
   [splashScreen orderOut : nil];
   [splashScreen release];
   splashScreen = nil;

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
      NSEvent * const timerEvent = [NSEvent otherEventWithType : NSApplicationDefined location : NSMakePoint(0, 0) modifierFlags : 0
                                    timestamp: 0. windowNumber : 0 context : nil subtype : 0 data1 : kSignalTimer data2 : 0];
      [NSApp postEvent : timerEvent atStart : NO];
   } else {
      //Scroll animation event.
   }
}

}//extern "C"

namespace {

//_________________________________________________________________
bool InitCocoa()
{
   if (!topLevelPool) {
      [[NSApplication sharedApplication] setActivationPolicy : NSApplicationActivationPolicyAccessory];
      [[NSApplication sharedApplication] activateIgnoringOtherApps : YES];

      topLevelPool = [[NSAutoreleasePool alloc] init];
   }
   
   return true;
}

//_________________________________________________________________
bool InitTimers(bool background)
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

   if (!background) {
      //Scroll animation.
      CFScopeGuard<CFRunLoopTimerRef> guard2(CFRunLoopTimerCreate(kCFAllocatorDefault,
                                                                  CFAbsoluteTimeGetCurrent() + splashScreenDelayInSec,
                                                                  scrollInterval,
                                                                  0,
                                                                  0,
                                                                  ROOTSplashscreenTimerCallback,
                                                                  0
                                                                  ));

      if (!guard2.Get())
         return false;

      signalTimer = guard2.Get();
      guard2.Release();
   }
   
   //TODO: refactor CFScopeGuard
   scrollTimer = guard1.Get();
   guard1.Release();
   
   return true;
}

//_________________________________________________________________
void AttachTimers(bool background)
{
   assert(signalTimer != 0 && "AttachTimer, invalid signalTimer (null)");
   
   CFRunLoopAddTimer(CFRunLoopGetMain(), signalTimer, kCFRunLoopCommonModes);

   if (!background) {
      //We also have to scroll.
      assert(scrollTimer != 0 && "AttachTimer, invalid scrollTimer (null)");
      CFRunLoopAddTimer(CFRunLoopGetMain(), scrollTimer, kCFRunLoopCommonModes);
   }
}

//_________________________________________________________________
void RemoveTimers(bool background)
{
   assert(signalTimer != 0 && "RemoveTimers, signalTimer is null");
   
   CFRunLoopRemoveTimer(CFRunLoopGetMain(), signalTimer, kCFRunLoopCommonModes);
   CFRunLoopTimerInvalidate(signalTimer);
   //TODO: test if I also have to call release!!!
   signalTimer = 0;
   
   if (!background) {
      assert(scrollTimer != 0 && "RemoveTimers, scrollTimer is null");
      CFRunLoopRemoveTimer(CFRunLoopGetMain(), signalTimer, kCFRunLoopCommonModes);
      CFRunLoopTimerInvalidate(signalTimer);
      //TODO: test if I also have to call release!!!
      signalTimer = 0;
      
   }
}

//_________________________________________________________________
void ProcessScrollTimerEvent(NSEvent *event)
{
//   assert(event != nil && "ProcessTimerEvent, parameter 'event' is nil");
#pragma unused(event)
   //TODO: scroll event.
}

//_________________________________________________________________
void RunEventLoop()
{
   //Kind of event loop.
   
   if (!InitTimers(false))//false == foreground.
      return;
   
   AttachTimers(false);//false == foreground.
   
   popupDone = false;

   while (!popupDone) {
      //Here we (possibly) suspend waiting for event.
      if (NSEvent * const event = [NSApp nextEventMatchingMask : NSAnyEventMask untilDate : [NSDate distantFuture] inMode : NSDefaultRunLoopMode dequeue : YES]) {
         //Let's first check the type:
         if (event.type == NSApplicationDefined) {//One of our timers 'fired'.
            if (event.data1 == kSignalTimer)
               popupDone = !showAboutInfo && !StayUp() && popdown;
            else
               ProcessScrollTimerEvent(event);
         } else
            [NSApp sendEvent : event];
      }
   }
   
   RemoveTimers(false);//false == foreground.
   //Empty the queue (hehehe, this makes me feel ... uneasy :) ).
   while ([NSApp nextEventMatchingMask : NSAnyEventMask untilDate : nil inMode : NSDefaultRunLoopMode dequeue : YES]);
}

//_________________________________________________________________
void WaitChildGeneric()
{
   //Wait till child (i.e. ROOT) is finished. From rootx.cxx.
   /*
   using ROOT::ROOTX::gChildpid;
   
   int status = 0;

   do {
      while (::waitpid(gChildpid, &status, WUNTRACED) < 0) {
         if (errno != EINTR)
            break;
         errno = 0;
      }

      if (WIFEXITED(status))
         std::exit(WEXITSTATUS(status));

      if (WIFSIGNALED(status))
         std::exit(WTERMSIG(status));

      if (WIFSTOPPED(status)) {         // child got ctlr-Z
         ::raise(SIGTSTP);                // stop also parent
         ::kill(gChildpid, SIGCONT);       // if parent wakes up, wake up child
      }
   } while (WIFSTOPPED(status));

   std::exit(0);
   */
}

//


//_________________________________________________________________
void RunEventLoopInBackground()
{
   if (!InitTimers(true))//true == background
      //This is actually something really fatal! Try to 'roll-back to X11 version'.
      return WaitChildGeneric();
   
   AttachTimers(true);//only the 'signal' timer.
   
   while (true) {

   }
}

//_________________________________________________________________
bool StayUp()
{
   //Taken from rootxx.cxx.
   const int splashScreenDelay = int(splashScreenDelayInSec * 1000);

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

//_________________________________________________________________
bool CreateSplashscreen()
{
   //Try to create NSImage out of Splash.gif, create NSPanel
   //with ROOTSplashscreenView as its content view + our background image.

   //0. Image for splash screen's background.
#ifdef ROOTICONPATH
   const std::string fileName(std::string(ROOTICONPATH) + "/Splash.gif");
#else
   const char * const env = std::getenv("ROOTSYS");
   if (!env) {
      //TODO: diagnostic.
      return false;
   }
   
   const std::string fileName(std::string(env) + "/icons/Splash.gif");
#endif

   using ROOT::MacOSX::Util::NSScopeGuard;

   const NSScopeGuard<NSString> nsStringGuard([[NSString alloc] initWithFormat : @"%s", fileName.c_str()]);
   if (!nsStringGuard.Get()) {
      //TODO: diagnostic.
      return false;
   }
   
   const NSScopeGuard<NSImage> imageGuard([[NSImage alloc] initWithContentsOfFile : nsStringGuard.Get()]);
   if (!imageGuard.Get()) {
      //TODO: diagnostic.
      return false;
   }
   
   const CGSize imageSize = imageGuard.Get().size;
   //These sizes are from X11 version, they are related to the geometry of a scroll view.
   if (imageSize.width < 300 || imageSize.height < 285) {
      //TODO: diagnostic.
      return false;
   }
   
   //1. Splash-screen ('panel' + its content view).
   NSScopeGuard<ROOTSplashScreenPanel> splashGuard([[ROOTSplashScreenPanel alloc]
                                                    initWithContentRect : CGRectMake(0, 0, imageSize.width, imageSize.height)
                                                    styleMask : NSBorderlessWindowMask
                                                    backing : NSBackingStoreBuffered
                                                    defer : NO]);
   if (!splashGuard.Get()) {
      //TODO: diagnostic.
      return false;
   }

   const NSScopeGuard<ROOTSplashScreenView> viewGuard([[ROOTSplashScreenView alloc] initWithImage : imageGuard.Get()]);
   if (!viewGuard.Get()) {
      //TODO: diagnostic.
      return false;
   }
   
   [splashGuard.Get() setContentView : viewGuard.Get()];
   splashScreen = splashGuard.Get();
   splashGuard.Release();
   
   return true;
}

//_________________________________________________________________
void SetSplashscreenPosition()
{
   assert(splashScreen != nil && "SetSplashscreenPosition, splashScreen is nil");
   
   //Set the splash-screen's position (can it be wrong for a multi-head setup?)
   //TODO: check with a secondary display.
   if (NSScreen * const screen = [NSScreen mainScreen]) {
      const NSRect screenFrame = screen.frame;
      const CGSize splashSize = splashScreen.frame.size;

      const CGPoint origin = CGPointMake(screenFrame.origin.x + screenFrame.size.width / 2 - splashSize.width / 2,
                                         screenFrame.origin.y + screenFrame.size.height / 2 - splashSize.height / 2);

      [splashScreen setFrameOrigin : origin];
   }//else - is it possible? TODO: diagnostic.
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

}//unnamed namespace.


namespace ROOT {
namespace ROOTX {

//This is a 'replacement' version.
//_________________________________________________________________
void WaitChild()
{
   RunEventLoopInBackground();
}

}
}
