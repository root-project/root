#import <UIKit/UIKit.h>

@interface LineWidthCell : UIView {
@private
   CGFloat lineWidth;
}

- (id) initWithFrame : (CGRect) frame width : (CGFloat) lineWidth;
- (void) setLineWidth : (CGFloat)width;

@end
