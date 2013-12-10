#import <UIKit/UIKit.h>

//Starting from iOS 6 we have a nice UICollectionView/UICollectionViewController.
//This class was born in iOS 4 times, so it's a kind of ugly and primitive collection view.
//Still I do not see any need in replacing it with a UICollectionViewController - it works as it is.

@class FileShortcutView;

@interface FileCollectionViewController : UIViewController <UINavigationControllerDelegate, UINavigationBarDelegate,
                                                            UIActionSheetDelegate, UIGestureRecognizerDelegate>

- (void) fileWasSelected : (FileShortcutView*) shortcut;
- (void) tryToDelete : (FileShortcutView*) shortcut;
- (void) addRootFile : (NSString *) fileName;

@end
