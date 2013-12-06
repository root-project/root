#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface SelectionView : UIView

@property (nonatomic) BOOL panActive;
@property (nonatomic) CGPoint panStart;
@property (nonatomic) CGPoint currentPanPoint;
@property (nonatomic) BOOL verticalPanDirection;

- (id)initWithFrame : (CGRect) frame withPad : (ROOT::iOS::Pad *) p;
- (void) setPad : (ROOT::iOS::Pad *) p;

@end
