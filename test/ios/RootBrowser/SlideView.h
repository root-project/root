#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface SlideView : UIView {
   ROOT::iOS::Pad *pad;
}

+ (CGSize) slideSize;
+ (CGRect) slideFrame;

- (id) initWithFrame : (CGRect)rect;
- (void) setPad : (ROOT::iOS::Pad *)pad;

@end
