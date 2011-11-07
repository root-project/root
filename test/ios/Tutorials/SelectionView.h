#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface SelectionView : UIView

- (void) setShowRotation : (BOOL) show;
- (void) setEvent : (int) ev atX : (int) x andY : (int) y;
- (void) setPad : (ROOT::iOS::Pad *)pad;

@end
