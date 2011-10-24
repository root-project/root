#import <UIKit/UIKit.h>

@interface ColorCell : UIView {
@private
   float rgb[3];
}

- (id) initWithFrame : (CGRect) frame;
- (void) dealloc;

- (void) setRGB : (const double *) rgb;
- (void) drawRect : (CGRect) rect;

+ (CGFloat) cellAlpha;

@end
