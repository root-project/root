#import <UIKit/UIKit.h>

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadImageScrollView : UIScrollView <UIScrollViewDelegate>

@property (nonatomic, retain) UIImage *padImage;

+ (CGRect) defaultImageFrame;

- (id) initWithFrame : (CGRect)frame;
- (void) setPad : (ROOT::iOS::Pad *)pad;
- (void) setPad : (ROOT::iOS::Pad *)pad andImage : (UIImage *)image;
- (void) resetToFrame : (CGRect)frame;

@end
