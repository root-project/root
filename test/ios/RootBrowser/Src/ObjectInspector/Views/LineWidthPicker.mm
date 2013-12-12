#import <cassert>

#import "LineWidthPicker.h"
#import "LineWidthCell.h"

@implementation LineWidthPicker {
   LineWidthCell *lineWidthView;
   UIImage *backgroundImage;
}

//____________________________________________________________________________________________________
- (void) lateInit
{
   // Initialization code
   lineWidthView = [[LineWidthCell alloc] initWithFrame : CGRectMake(10.f, 10.f, 120.f, 50.f) width : 1.f];
   [self addSubview : lineWidthView];
      
   backgroundImage = [UIImage imageNamed : @"line_width_bkn.png"];
}

//____________________________________________________________________________________________________
- (instancetype) initWithFrame : (CGRect) frame
{
   if (self = [super initWithFrame : frame])
      [self lateInit];

   return self;
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect) rect
{
   if (!backgroundImage)
      [self lateInit];

   [backgroundImage drawInRect : rect];
}

//____________________________________________________________________________________________________
- (void) setLineWidth : (float) width
{
   assert(width > 0.f && "setLineWidth:, parameter 'width' must be a positive number");

   [lineWidthView setLineWidth : width];
   [lineWidthView setNeedsDisplay];
}

@end
