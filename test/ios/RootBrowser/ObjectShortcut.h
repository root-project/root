#import <UIKit/UIKit.h>

@class FileContentController;

@interface ObjectShortcut : UIView

@property (nonatomic, assign) BOOL isDirectory;
@property (nonatomic, retain) UIImage *icon;
@property (nonatomic, readonly) unsigned objectIndex;
@property (nonatomic, retain) NSString *objectName;

+ (CGFloat) iconWidth;
+ (CGFloat) iconHeight;
+ (CGFloat) textHeight;
+ (CGRect) defaultRect;


- (id) initWithFrame : (CGRect)frame controller : (FileContentController*) c forObjectAtIndex : (unsigned)objIndex withThumbnail : (UIImage *)thumbnail;
- (id) initWithFrame : (CGRect)frame controller : (FileContentController*) c forFolderAtIndex : (unsigned)index;

@end
