#import <CoreGraphics/CGContext.h>
#import <QuartzCore/QuartzCore.h>

//C++ (ROOT) imports.
#import "IOSFillPatterns.h"

#import "PatternCell.h"

@implementation PatternCell {
   unsigned patternIndex;
   BOOL solid;
}

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame andPattern : (unsigned) index
{
   self = [super initWithFrame : frame];

   if (self) {
      patternIndex = index;
      solid = NO;

      self.opaque = NO;
      self.layer.shadowColor = [UIColor darkGrayColor].CGColor;
      self.layer.shadowOpacity = 0.4f;
      self.layer.shadowOffset = CGSizeMake(4.f, 4.f);

      //Shadows are quite expensive if path is not specified.
      self.layer.shadowPath = [UIBezierPath bezierPathWithRect : CGRectMake(10.f, 10.f, frame.size.width - 20.f, frame.size.height - 20.f)].CGPath;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) setAsSolid
{
   solid = YES;
}

//____________________________________________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   rect.origin.x = 10.f;
   rect.origin.y = 10.f;
   rect.size.width -= 20.f;
   rect.size.height -= 20.f;

   CGContextRef ctx = UIGraphicsGetCurrentContext();

   //Fill view with pattern.
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, rect);

   if (!solid) { //Solid fill - no pattern.
      float rgb[] = {0.f, 0.f, 0.f};
      CGPatternRef pattern = ROOT::iOS::GraphicUtils::gPatternGenerators[patternIndex](rgb);

      CGColorSpaceRef colorSpace = CGColorSpaceCreatePattern(0);
      const float alpha = 1.f;

      CGContextSetFillColorSpace(ctx, colorSpace);
      CGContextSetFillPattern(ctx, pattern, &alpha);
      CGContextFillRect(ctx, rect);

      CGColorSpaceRelease(colorSpace);
      CGPatternRelease(pattern);
   }
}

@end
