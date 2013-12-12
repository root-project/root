#import <cassert>

#import <QuartzCore/QuartzCore.h>

#import "LineWidthCell.h"

@implementation LineWidthCell {
   CGFloat lineWidth;
}

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame width : (CGFloat) w
{
   assert(w >= 0.f && "initWithFrame:width:, parameter 'w' is negative");

   if (self = [super initWithFrame : frame]) {
      lineWidth = w;
      
      self.layer.shadowOpacity = 0.4f;
      self.layer.shadowColor = [UIColor darkGrayColor].CGColor;
      self.layer.shadowOffset = CGSizeMake(4.f, 4.f);

      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect) rect
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
   
   NSString * const label = [NSString stringWithFormat : @"(%d)", (int)lineWidth];
   NSDictionary * const attributes = @{NSFontAttributeName : [UIFont systemFontOfSize : 10],
                                       NSForegroundColorAttributeName : [UIColor blackColor]};
   [label drawInRect : CGRectMake(rect.size.width / 2 - 10.f, rect.size.height / 2 - 15.f, 40.f, 60.f) withAttributes : attributes];
}

//____________________________________________________________________________________________________
- (void) setLineWidth : (CGFloat) width
{
   assert(lineWidth > 0.f && "setLineWidth:, parameter 'width' must be a positive number");

   lineWidth = width;
}

@end
