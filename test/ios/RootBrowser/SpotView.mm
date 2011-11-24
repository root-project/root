#import <CoreGraphics/CGContext.h>

#import "SpotView.h"

@implementation SpotView

- (id)initWithFrame:(CGRect)frame
{
   if ([super initWithFrame : frame]) {
      self.alpha = 0.f;
      self.multipleTouchEnabled = NO;
      self.backgroundColor = [UIColor orangeColor];
   }

   return self;
}

@end
