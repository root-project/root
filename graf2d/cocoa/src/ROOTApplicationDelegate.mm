#import "ROOTApplicationDelegate.h"
#import "QuartzWindow.h"
#import "CocoaUtils.h"
#import "TGWindow.h"
#import "TGClient.h"

@implementation ROOTApplicationDelegate {
   NSMutableArray *fWindowStack;
}

//______________________________________________________________________________
- (id) init
{
   if (self = [super init]) {
      [NSApp setDelegate : self];
      fWindowStack = [[NSMutableArray alloc] init];
   }
   
   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   [NSApp setDelegate : nil];//?
   [fWindowStack release];
   [super dealloc];
}


//NSApplicationDelegate.

//______________________________________________________________________________
- (void) applicationWillResignActive : (NSNotification *) aNotification
{
   //Popup windows, menus, color-selectors, etc. - they all have
   //a problem: due to some reason, Cocoa changes the z-stack order
   //of such a window while switching between applications (using alt-tab, for example).
   //This leads to a very annoying effect: you open a menu, alt-tab,
   //alt-tab back and ... popup or menu is now behind the main window.
   //I have to save/restore this z-stack order here.

   (void) aNotification;

   if (!gClient) {
      NSLog(@"ROOTApplicationDelegate: -applicationWillResignActive:, gClient is null");
      return;
   }

   const ROOT::MacOSX::Util::AutoreleasePool pool;

   NSArray * const orderedWindows = [NSApp orderedWindows];

   [fWindowStack removeAllObjects];
   for (NSWindow *nsWindow in orderedWindows) {
      if (![nsWindow isKindOfClass : [QuartzWindow class]])
         continue;

      QuartzWindow *qWindow = (QuartzWindow *)nsWindow;

      if (TGWindow *rootWindow = gClient->GetWindowById(qWindow.fID)) {
         if (!rootWindow->InheritsFrom("TGToolTip")) {
            if (qWindow.fMapState == kIsViewable)
               [fWindowStack addObject : nsWindow];
         }
      }
   }
}

//______________________________________________________________________________
- (void) applicationDidBecomeActive : (NSNotification *) aNotification
{
   //Popup windows, menus, color-selectors, etc. - they all have
   //a problem: due to some reason, Cocoa changes the z-stack order
   //of such a window while switching between applications (using alt-tab, for example).
   //This leads to a very annoying effect: you open a menu, alt-tab,
   //alt-tab back and ... popup or menu is now behind the main window.
   //I have to save/restore this z-stack order here.

   (void) aNotification;

   if (![fWindowStack count])
      return;

   const ROOT::MacOSX::Util::AutoreleasePool pool;

   NSEnumerator * const reverseEnumerator = [fWindowStack reverseObjectEnumerator];
   for (QuartzWindow *qw in reverseEnumerator)
      [qw mapRaised];

   [fWindowStack removeAllObjects];
}

@end
