#import <UIKit/UIKit.h>

//
//PadSlideView is a small special pad view to use in a "slideshow" animation.
//Does not support any object picking/editing, just a surface to render and
//a view to animate a transition.
//

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadSlideView : UIView

+ (CGSize) slideSize;
+ (CGRect) slideFrame;

- (instancetype) initWithFrame : (CGRect) rect;
- (void) setPad : (ROOT::iOS::Pad *) pad;

@end
