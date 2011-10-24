#import <UIKit/UIKit.h>

@interface PadImageView : UIView {
@private
   UIImage *padImage;
   BOOL zoomed;
}

@property (assign) BOOL zoomed;
@property (nonatomic, retain) UIImage *padImage;

- (void) setPadImage : (UIImage *)image;
- (UIImage *) padImage;

@end
