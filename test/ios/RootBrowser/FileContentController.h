#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@class ObjectShortcut;

@interface FileContentController : UIViewController <UITableViewDataSource> {
@private
   ROOT::iOS::Browser::FileContainer *fileContainer;
   __weak IBOutlet UIScrollView *scrollView;
}

//@property (nonatomic, assign) id<UITableViewDataSource> *
@property (nonatomic, readonly) ROOT::iOS::Browser::FileContainer *fileContainer;

- (void) activateForFile : (ROOT::iOS::Browser::FileContainer *)container;
- (void) selectObjectFromFile : (ObjectShortcut *)obj;

@end
