#import <UIKit/UIKit.h>

@class RootFileController;

class TApplication;

@interface root_browserAppDelegate : NSObject <UIApplicationDelegate> {
   TApplication *rootApp;

   RootFileController *rootController;
   UINavigationController *navigationController;
}

@property (nonatomic, retain) IBOutlet UIWindow *window;

@end
