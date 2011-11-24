#import "PadImageView.h"

@implementation PadImageView

@synthesize padImage;
@synthesize zoomed;

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   if (padImage)
      [padImage drawInRect : rect];
}

@end
