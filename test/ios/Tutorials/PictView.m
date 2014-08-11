#import <QuartzCore/QuartzCore.h>

#import "PictView.h"

@implementation PictView

//_________________________________________________________________
- (id) initWithFrame : (CGRect)frame andIcon : (NSString *)iconName
{
   //self = [super initWithFrame:frame];

   self = [super initWithImage : [UIImage imageNamed : iconName]];

   if (self) {
      self.frame = frame;
      //View is transparent with shadow (under non-transparent pixels in a picture).
      self.opaque = NO;
      self.alpha = 0.5f;
      self.layer.shadowColor = [UIColor blackColor].CGColor;
      self.layer.shadowOpacity = 0.7f;
      self.layer.shadowOffset = CGSizeMake(3.f, 3.f);
      self.userInteractionEnabled = YES;
   }

   return self;
}

@end
