#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface SelectionView : UIView

@property (nonatomic, assign) BOOL panActive;
@property (nonatomic, assign) CGPoint panStart;
@property (nonatomic, assign) CGPoint currentPanPoint;
@property (nonatomic, assign) BOOL verticalPanDirection;

- (id)initWithFrame : (CGRect)frame withPad : (ROOT::iOS::Pad *)p;
- (void) setPad : (ROOT::iOS::Pad *)p;

@end
