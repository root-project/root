#import "ROOTApplicationDelegate.h"

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
   (void) aNotification;
   //NSLog(@"applicationWillResignActive");
}

//______________________________________________________________________________
- (void) applicationDidBecomeActive : (NSNotification *) aNotification
{
   (void) aNotification;
   //NSLog(@"applicationDidBecomeActive");
}

@end
