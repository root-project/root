#import <UIKit/UIKit.h>

@class FileShortcutView;

@interface FileCollectionViewController : UIViewController <UINavigationControllerDelegate, UINavigationBarDelegate, UIActionSheetDelegate>

- (void) fileWasSelected : (FileShortcutView*) shortcut;
- (void) tryToDelete : (FileShortcutView*) shortcut;
- (void) addRootFile : (NSString *) fileName;

@end
