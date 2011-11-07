#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface SlideView : UIView

+ (CGSize) slideSize;
+ (CGRect) slideFrame;

- (id) initWithFrame : (CGRect)rect;
- (void) setPad : (ROOT::iOS::Pad *)pad;

@end
