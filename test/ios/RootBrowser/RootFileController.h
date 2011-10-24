#import <UIKit/UIKit.h>

@class FileContentController;
@class FileShortcut;

@interface RootFileController : UIViewController <UINavigationControllerDelegate, UINavigationBarDelegate> {
@private
   NSMutableArray *fileContainers;

   IBOutlet UIScrollView *scrollView;
   IBOutlet UIView *fileOpenView;
   IBOutlet UITextField *fileNameField;
}

@property (nonatomic, retain) UIScrollView *scrollView;
@property (nonatomic, retain) UIView *fileOpenView;
@property (nonatomic, retain) UITextField *fileNameField;

- (void) fileWasSelected : (FileShortcut*) shortcut;
- (void) addFileShortcut : (NSString *) fileName;
- (void) hideFileOpenView;

- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;

@end
