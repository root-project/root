#import <UIKit/UIKit.h>

//
//PadSelectionView is a special transparent view on top of a pad view,
//rendering the picked object in a special way.
//

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadSelectionView : UIView

@property (nonatomic) BOOL panActive;
@property (nonatomic) CGPoint panStart;
@property (nonatomic) CGPoint currentPanPoint;
@property (nonatomic) BOOL verticalPanDirection;

- (instancetype) initWithFrame : (CGRect) frame withPad : (ROOT::iOS::Pad *) p;
- (void) setPad : (ROOT::iOS::Pad *) p;

@end
