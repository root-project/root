#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Painter;
class Pad;

}
}

@class PadView;

@interface SelectionView : UIView {
   BOOL showRotation;
   int ev;
   int px;
   int py;
   ROOT::iOS::Pad *pad;

   PadView *view;
}

- (void) setShowRotation : (BOOL) show;
- (void) setEvent : (int) ev atX : (int) x andY : (int) y;
- (void) setPad : (ROOT::iOS::Pad *)pad;

@end
