#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>

#import "LineWidthCell.h"


@implementation LineWidthCell {
   CGFloat lineWidth;
}

//____________________________________________________________________________________________________
- (id)initWithFrame:(CGRect)frame width : (CGFloat)w
{
   self = [super initWithFrame:frame];

   if (self) {
      lineWidth = w;

      self.layer.shadowOpacity = 0.4f;
      self.layer.shadowColor = [UIColor darkGrayColor].CGColor;
      self.layer.shadowOffset = CGSizeMake(4.f, 4.f);

      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();

   //Draw the line.
   CGContextSetLineCap(ctx, kCGLineCapRound);
   CGContextSetLineWidth(ctx, lineWidth);
   CGContextSetRGBStrokeColor(ctx, 0.f, 0.f, 0.f, 1.f);
   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, 10.f, rect.size.height / 2);
   CGContextAddLineToPoint(ctx, rect.size.width - 10, rect.size.height / 2);
   CGContextStrokePath(ctx);

   NSString *label = [NSString stringWithFormat:@"(%d)", (int)lineWidth];
   CGContextSetRGBFillColor(ctx, 0.f, 0.f, 1.f, 1.f);
   [label drawInRect:CGRectMake(rect.size.width / 2 - 10.f, rect.size.height / 2 - 15.f, 40.f, 60.f) withFont : [UIFont systemFontOfSize : 10]];
}

//____________________________________________________________________________________________________
- (void) setLineWidth : (CGFloat)width
{
   lineWidth = width;
}

@end
