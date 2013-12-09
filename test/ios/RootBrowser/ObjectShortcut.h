#import <UIKit/UIKit.h>

@class FileContentViewController;
@class SpotObjectView;

@interface ObjectShortcut : UIView

@property (nonatomic, readonly) SpotObjectView *spot;
@property (nonatomic, readonly) BOOL isDirectory;
@property (nonatomic, readonly) unsigned objectIndex;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;
+ (CGFloat) textHeight;
+ (CGRect) defaultRect;

- (id) initWithFrame : (CGRect) frame controller : (FileContentViewController *) c forObjectAtIndex : (unsigned) objIndex withThumbnail : (UIImage *) thumbnail;
- (id) initWithFrame : (CGRect) frame controller : (FileContentViewController *) c forFolderAtIndex : (unsigned) index;

@end
