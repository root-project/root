#import <UIKit/UIKit.h>

@interface LineStyleCell : UIView {
@private
   unsigned lineStyle;
}

- (id) initWithFrame : (CGRect)frame lineStyle : (unsigned) style;

@end
