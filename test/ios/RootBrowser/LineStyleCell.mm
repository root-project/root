#import <QuartzCore/QuartzCore.h>
#import <CoreGraphics/CGContext.h>

#import "LineStyleCell.h"

//C++ (ROOT) imports.
#import "IOSLineStyles.h"

@implementation LineStyleCell

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame lineStyle : (unsigned) style
{
   self = [super initWithFrame:frame];

   if (self) {
      lineStyle = style;
      
      self.layer.shadowOffset = CGSizeMake(4.f, 4.f);
      self.layer.shadowOpacity = 0.7f;
      self.layer.shadowColor = [UIColor darkGrayColor].CGColor;
      
      self.opaque = NO;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   [super dealloc];
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   
   CGContextSetRGBStrokeColor(ctx, 0.3f, 0.3f, 0.3f, 0.4f);
   CGContextStrokeRect(ctx, rect);

   CGContextSetRGBStrokeColor(ctx, 0.f, 0.f, 0.f, 1.f);

   if (lineStyle > 1 && lineStyle <= 10)
      CGContextSetLineDash(ctx, 0., ROOT::iOS::GraphicUtils::dashLinePatterns[lineStyle - 1], ROOT::iOS::GraphicUtils::linePatternLengths[lineStyle - 1]);
   else
      CGContextSetLineDash(ctx, 0., 0, 0);
   
   CGContextSetLineWidth(ctx, 2.f);
   
   CGContextBeginPath(ctx);
   CGContextMoveToPoint(ctx, 10.f, rect.size.height  - 10.f);
   CGContextAddLineToPoint(ctx, rect.size.width - 10, 10.f);

   CGContextStrokePath(ctx);
}

@end
