#import "ROOTApplicationDelegate.h"
#import "TApplication.h"

@implementation ROOTApplicationDelegate

//______________________________________________________________________________
- (id) init
{
   if (self = [super init]) {
      [NSApp setDelegate : self];
   }
   
   return self;
}

//______________________________________________________________________________
- (void) dealloc
{
   [NSApp setDelegate : nil];//?
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

   //Popups were fixed using transient hint, noop now.
   (void) aNotification;
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

   //Popups were fixed using transient hint, noop now.
   (void) aNotification;
}

//______________________________________________________________________________
- (void) quitROOT
{
   gApplication->Terminate(0);
}

//______________________________________________________________________________
- (NSApplicationTerminateReply) applicationShouldTerminate : (NSApplication *) sender
{
   (void) sender;
   [self performSelector : @selector(quitROOT) withObject : nil afterDelay : 0.1];
   return NSTerminateCancel;
}


@end
