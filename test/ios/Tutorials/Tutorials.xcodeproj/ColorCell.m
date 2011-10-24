#import <CoreGraphics/CoreGraphics.h>

#import "ColorCell.h"


@implementation ColorCell

//______________________________________________________________________________
- (id)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    if (self) {
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
   CGContextSetRGBFillColor(ctx, rgb[0], rgb[1], rgb[2], 0.8);
   CGContextFillRect(ctx, rect);
}

- (void)dealloc
{
   [super dealloc];
}

@end
