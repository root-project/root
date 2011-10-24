#import "LineWidthPicker.h"
#import "LineWidthCell.h"

@implementation LineWidthPicker

//____________________________________________________________________________________________________
- (void) lateInit
{
   // Initialization code
   lineWidthView = [[LineWidthCell alloc] initWithFrame : CGRectMake(10.f, 10.f, 120.f, 50.f) width : 1.f];
   [self addSubview : lineWidthView];
   [lineWidthView release];
      
   backgroundImage = [UIImage imageNamed:@"line_width_bkn.png"];
   [backgroundImage retain];
}

//____________________________________________________________________________________________________
- (id)initWithFrame:(CGRect)frame
{
   self = [super initWithFrame:frame];
   if (self) {
      [self lateInit];
   }
   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   [backgroundImage release];
   [super dealloc];
}

//____________________________________________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   if (!backgroundImage)
      [self lateInit];

   [backgroundImage drawInRect : rect];
}

//____________________________________________________________________________________________________
- (void) setLineWidth : (float)width
{
   [lineWidthView setLineWidth : width];
   [lineWidthView setNeedsDisplay];
}

@end
