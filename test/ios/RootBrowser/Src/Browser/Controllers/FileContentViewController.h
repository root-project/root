#import <UIKit/UIKit.h>

#import "SearchViewController.h"

//
//These view/controller classes are a legacy code and implement
//what is essentially UICollectionView/UICollectionViewController today
//with some additional features.
//As soon as it works as I want, I do not see any reason
//to get rid of it.
//

@class FileContainerElement;
@class ObjectShortcutView;

namespace ROOT {
namespace iOS {
namespace Browser {

class FileContainer;

}
}
}

@interface FileContentViewController : UIViewController <UISearchBarDelegate, UIPopoverControllerDelegate, SearchViewDelegate>

@property (nonatomic, readonly) ROOT::iOS::Browser::FileContainer *fileContainer;

- (void) activateForFile : (ROOT::iOS::Browser::FileContainer *) container;

- (void) selectObjectFromFile : (ObjectShortcutView *) obj;
- (void) searchController : (SearchViewController *) controller didSelectKey : (FileContainerElement *) key;

@end
