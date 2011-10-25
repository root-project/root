#import <UIKit/UIKit.h>

@class SlideView;

namespace ROOT {
namespace iOS {

class FileContainer;
class Pad;

}
}

@interface SlideshowController : UIViewController <UIScrollViewDelegate> {
@private
   SlideView *padViews[2];//The current and the next in a slide show.

   unsigned visiblePad;
   unsigned nCurrentObject;
   
   ROOT::iOS::FileContainer *fileContainer;
   
   NSTimer *timer;
   IBOutlet UIView *parentView;
   IBOutlet UIView *padParentView;
}

@property (nonatomic, retain) UIView *parentView;
@property (nonatomic, retain) UIView *padParentView;

- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil fileContainer : (ROOT::iOS::FileContainer *)container;

@end
