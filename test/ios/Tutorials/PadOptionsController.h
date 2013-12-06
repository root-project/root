#import <CoreGraphics/CGPattern.h>
#import <UIKit/UIKit.h>

@class PadView;

namespace ROOT {
namespace iOS {

class Pad;

}
}


@interface PadOptionsController : UIViewController

- (void) setView : (PadView *) view andPad : (ROOT::iOS::Pad *) pad;

@end
