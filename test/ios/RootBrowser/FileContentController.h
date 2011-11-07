#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class FileContainer;

}
}

@class ObjectShortcut;

@interface FileContentController : UIViewController {
@private
   ROOT::iOS::FileContainer *fileContainer;
   __weak IBOutlet UIScrollView *scrollView;
}

@property (nonatomic, readonly) ROOT::iOS::FileContainer *fileContainer;

- (void) activateForFile : (ROOT::iOS::FileContainer *)container;
- (void) selectObjectFromFile : (ObjectShortcut *)obj;

@end
