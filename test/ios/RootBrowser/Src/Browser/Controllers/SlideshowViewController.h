#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface SlideshowViewController : UIViewController <UIScrollViewDelegate>

- (id) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil fileContainer : (ROOT::iOS::Browser::FileContainer *) container;

@end
