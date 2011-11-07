#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class FileContainer;

}
}

@interface SlideshowController : UIViewController <UIScrollViewDelegate> {
@private
   __weak IBOutlet UIView *parentView;
   __weak IBOutlet UIView *padParentView;
}

- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil fileContainer : (ROOT::iOS::FileContainer *)container;

@end
