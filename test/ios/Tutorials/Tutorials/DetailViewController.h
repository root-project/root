#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

namespace Demos {
class DemoBase;
}

}
}

@interface DetailViewController : UIViewController <UIPopoverControllerDelegate, UISplitViewControllerDelegate, UIScrollViewDelegate> 

@property (nonatomic, retain) IBOutlet UITabBar *tabBar;
@property (nonatomic, retain) IBOutlet UIToolbar *toolbar;
@property (nonatomic, retain) IBOutlet UIView *help;

- (void) setActiveDemo : (ROOT::iOS::Demos::DemoBase *)demo;
- (void) dismissPopover;

- (IBAction) zoomButtonPressed;
- (IBAction) editButtonPressed : (id) sender;
- (IBAction) selectButtonPressed;
- (IBAction) showHelp;

@end
