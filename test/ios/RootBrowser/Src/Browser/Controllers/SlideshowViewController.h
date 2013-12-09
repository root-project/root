#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface SlideshowViewController : UIViewController <UIScrollViewDelegate>

- (void) setFileContainer : (ROOT::iOS::Browser::FileContainer *) container;

@end
