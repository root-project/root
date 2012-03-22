#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadImageScrollView : UIScrollView <UIScrollViewDelegate>

+ (CGRect) defaultImageFrame;

- (id) initWithFrame : (CGRect)frame;

- (void) setPad : (ROOT::iOS::Pad *)pad;
- (void) resetToFrame : (CGRect)frame;

@end
