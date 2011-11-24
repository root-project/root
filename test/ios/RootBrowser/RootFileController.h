#import <UIKit/UIKit.h>

@class FileShortcut;

@interface RootFileController : UIViewController <UINavigationControllerDelegate, UINavigationBarDelegate> {
@private
   __weak IBOutlet UIScrollView *scrollView;
   __weak IBOutlet UIView *fileOpenView;
   __weak IBOutlet UITextField *fileNameField;
}

- (void) fileWasSelected : (FileShortcut*)shortcut;
- (void) tryToDelete : (FileShortcut*)shortcut;
- (void) addRootFile : (NSString *)fileName;

- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;

@end
