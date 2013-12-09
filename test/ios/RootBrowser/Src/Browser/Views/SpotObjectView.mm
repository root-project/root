#import "SpotObjectView.h"

@implementation SpotObjectView

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect) frame
{
   if (self = [super initWithFrame : frame]) {
      self.alpha = 0.f;
      self.multipleTouchEnabled = NO;
      self.backgroundColor = [UIColor orangeColor];
   }

   return self;
}

@end
