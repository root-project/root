#import <UIKit/UIKit.h>

@class FileContentViewController;
@class SpotObjectView;

@interface ObjectShortcutView : UIView

@property (nonatomic, readonly) SpotObjectView *spot;
@property (nonatomic, readonly) BOOL isDirectory;
@property (nonatomic, readonly) unsigned objectIndex;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;
+ (CGFloat) textHeight;
+ (CGRect) defaultRect;

- (instancetype) initWithFrame : (CGRect) frame controller : (FileContentViewController *) c forObjectAtIndex : (unsigned) objIndex withThumbnail : (UIImage *) thumbnail;
- (instancetype) initWithFrame : (CGRect) frame controller : (FileContentViewController *) c forFolderAtIndex : (unsigned) index;

@end
