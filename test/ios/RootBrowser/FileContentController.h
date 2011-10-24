#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class FileContainer;

}
}

class TObject;

@class ObjectShortcut;

@interface FileContentController : UIViewController {
@private
   NSMutableArray *objectShortcuts;
   ROOT::iOS::FileContainer *fileContainer;
   IBOutlet UIScrollView *scrollView;
}

@property (nonatomic, retain) UIScrollView *scrollView;
@property (nonatomic, readonly) ROOT::iOS::FileContainer *fileContainer;

- (void) activateForFile : (ROOT::iOS::FileContainer *)container;
- (void) selectObjectFromFile : (ObjectShortcut *)obj;

@end
