#import <CoreGraphics/CoreGraphics.h>

#import "ColorCell.h"

@implementation ColorCell {
   double rgb[3];
}

//______________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   if (self = [super initWithFrame : frame]) {
      // Initialization code
      self.backgroundColor = [UIColor clearColor];
   }

   return self;
}

//______________________________________________________________________________
- (void) setRGB : (const double *) newRgb
{
   rgb[0] = newRgb[0];
   rgb[1] = newRgb[1];
   rgb[2] = newRgb[2];
}

//______________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextSetRGBFillColor(ctx, (CGFloat)rgb[0], (CGFloat)rgb[1], (CGFloat)rgb[2], 1.f);
   CGContextFillRect(ctx, rect);
}

@end
