#import <UIKit/UIKit.h>

#import "SearchController.h"

@class FileContainerElement;
@class ObjectShortcut;

namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface FileContentController : UIViewController <UISearchBarDelegate, UIPopoverControllerDelegate, SearchDelegate> {
@private
   ROOT::iOS::Browser::FileContainer *fileContainer;
   __weak IBOutlet UIScrollView *scrollView;
}

//@property (nonatomic, assign) id<UITableViewDataSource> *
@property (nonatomic, readonly) ROOT::iOS::Browser::FileContainer *fileContainer;

- (void) activateForFile : (ROOT::iOS::Browser::FileContainer *)container;
- (void) selectObjectFromFile : (ObjectShortcut *)obj;

- (void) searchesController : (SearchController *)controller didSelectKey:(FileContainerElement *)key;

@end
