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
   //Popups were fixed using transient hint, noop now.
   (void) aNotification;
}

//______________________________________________________________________________
- (void) applicationDidBecomeActive : (NSNotification *) aNotification
{
   //Popups were fixed using transient hint, noop now.
   (void) aNotification;
}

@end
