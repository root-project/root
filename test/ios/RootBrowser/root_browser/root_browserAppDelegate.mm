#import "root_browserAppDelegate.h"
#import "RootFileController.h"

#import "TApplication.h"

@implementation root_browserAppDelegate

@synthesize window=_window;

//____________________________________________________________________________________________________
- (void) initRootController
{
   rootController = [[RootFileController alloc] initWithNibName:@"RootFileController" bundle : nil];
   [rootController addFileShortcut : [[NSBundle mainBundle] pathForResource:@"demos" ofType:@"root"]];

   navigationController = [[UINavigationController alloc] initWithRootViewController : rootController];
   [rootController release];
   
   navigationController.navigationBar.barStyle = UIBarStyleBlackTranslucent;
   navigationController.delegate = rootController;
   
   [self.window addSubview : navigationController.view];
   [self.window makeKeyAndVisible];
}

//____________________________________________________________________________________________________
- (void) deleteRootController
{
   [navigationController.view removeFromSuperview];
   navigationController.delegate = nil;
   [navigationController release];
}

//____________________________________________________________________________________________________
- (BOOL)application:(UIApplication *)application didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
   // Override point for customization after application launch.
   rootApp = new TApplication("iosApp", 0, 0);
   return YES;
}

//____________________________________________________________________________________________________
- (void)applicationWillResignActive:(UIApplication *)application
{
   /*
    Sent when the application is about to move from active to inactive state. This can occur for certain types of temporary interruptions (such as an incoming phone call or SMS message) or when the user quits the application and it begins the transition to the background state.
    Use this method to pause ongoing tasks, disable timers, and throttle down OpenGL ES frame rates. Games should use this method to pause the game.
    */
//    NSLog(@"willResignActive");
}

//____________________________________________________________________________________________________
- (void)applicationDidEnterBackground:(UIApplication *)application
{
   /*
   Use this method to release shared resources, save user data, invalidate timers, and store enough application state information to restore your application to its current state in case it is terminated later. 
   If your application supports background execution, this method is called instead of applicationWillTerminate: when the user quits.
   */
//   NSLog(@"didEnterBackground");
   [self deleteRootController];
}

//____________________________________________________________________________________________________
- (void)applicationWillEnterForeground:(UIApplication *)application
{
   /*
   Called as part of the transition from the background to the inactive state; here you can undo many of the changes made on entering the background.
   */
//   NSLog(@"willEnterForeground");
}

//____________________________________________________________________________________________________
- (void)applicationDidBecomeActive:(UIApplication *)application
{
   /*
   Restart any tasks that were paused (or not yet started) while the application was inactive. If the application was previously in the background, optionally refresh the user interface.
   */
//   NSLog(@"didBecomeActive");
   [self initRootController];
}

//____________________________________________________________________________________________________
- (void)applicationWillTerminate:(UIApplication *)application
{
   /*
    Called when the application is about to terminate.
    Save data if appropriate.
    See also applicationDidEnterBackground:.
    */
 //  NSLog(@"applicationWillTerminate");
}

//____________________________________________________________________________________________________
- (void)dealloc
{
   [_window release];
   [super dealloc];
   delete rootApp;
}

@end
