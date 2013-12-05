#import <cassert>

#import "IOSFillPatterns.h"
#import "PatternCell.h"

@implementation PatternCell {
   unsigned patternIndex;
   BOOL solid;
}

//Pattern index must be in [0, kPredefinedFillPatterns) range +, probably,
//index is ignored if it's a 'solid' fill pattern.

//______________________________________________________________________________
- (id) initWithFrame : (CGRect) frame andPattern : (unsigned) index
{
   if (self = [super initWithFrame : frame]) {
      assert(index < ROOT::iOS::GraphicUtils::kPredefinedFillPatterns &&
             "initWithFrame:andPattern:, parameter 'index' is out of bounds");
   
      // Initialization code
      patternIndex = index;
      solid = NO;
   }

   return self;
}

//______________________________________________________________________________
- (void) setPattern : (unsigned) index
{
   assert(index < ROOT::iOS::GraphicUtils::kPredefinedFillPatterns &&
          "setPattern:, parameter 'index' is out of bounds");
   
   patternIndex = index;
   solid = NO;
}

//______________________________________________________________________________
- (void) setAsSolid
{
   solid = YES;
}

//______________________________________________________________________________
- (void) drawRect : (CGRect) rect
{
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
