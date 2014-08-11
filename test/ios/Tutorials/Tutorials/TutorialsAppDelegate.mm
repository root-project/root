#import "TutorialsAppDelegate.h"
#import "RootViewController.h"

#import "TApplication.h"

@implementation TutorialsAppDelegate {
   TApplication *rootApp;
}

@synthesize window;
@synthesize splitViewController;
@synthesize rootViewController;
@synthesize detailViewController;

//_________________________________________________________________
- (BOOL) application : (UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
   // Override point for customization after application launch.
   // Add the split view controller's view to the window and display.
   rootApp = new TApplication("iosApp", 0, 0);

   self.window.rootViewController = self.splitViewController;
   [self.window makeKeyAndVisible];

   return YES;
}

//_________________________________________________________________
- (void)applicationWillResignActive:(UIApplication *)application
{
   /*
    Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    Use this method to pause ongoing tasks, disable timers, and throttle down OpenGL ES frame rates. Games should use this method to pause the game.
    */
}

//_________________________________________________________________
- (void)applicationDidEnterBackground:(UIApplication *)application
{
   /*
    Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later.
    If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
    */
}

//_________________________________________________________________
- (void)applicationWillEnterForeground:(UIApplication *)application
{
   /*
    Called as part of the transition from the background to the inactive state; here you can undo many of the changes made on entering the background.
    */
}

//_________________________________________________________________
- (void)applicationDidBecomeActive:(UIApplication *)application
{
   /*
    Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
    */
}

//_________________________________________________________________
- (void)applicationWillTerminate:(UIApplication *)application
{
   /*
    Called when the application is about to terminate.
    Save data if appropriate.
    See also applicationDidEnterBackground:.
    */
}

//_________________________________________________________________
- (void)dealloc
{
   delete rootApp;
}

@end
