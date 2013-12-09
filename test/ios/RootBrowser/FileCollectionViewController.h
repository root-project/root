#import <UIKit/UIKit.h>

@class FileShortcut;

@interface FileCollectionViewController : UIViewController <UINavigationControllerDelegate, UINavigationBarDelegate, UIActionSheetDelegate>

- (void) fileWasSelected : (FileShortcut*) shortcut;
- (void) tryToDelete : (FileShortcut*) shortcut;
- (void) addRootFile : (NSString *) fileName;

@end
