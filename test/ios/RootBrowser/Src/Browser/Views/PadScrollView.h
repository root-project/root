#import <UIKit/UIKit.h>

//
//This is a special scroll view, containing
//a pad view, which can be zoomed and scrolled (if zoomed).
//
//

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadScrollView : UIScrollView <UIScrollViewDelegate>

+ (CGRect) defaultImageFrame;

- (instancetype) initWithFrame : (CGRect)frame;

- (void) setPad : (ROOT::iOS::Pad *) pad;
- (void) resetToFrame : (CGRect) frame;

@end
