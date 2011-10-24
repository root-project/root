#import <CoreGraphics/CoreGraphics.h>

#import "HintView.h"


@implementation HintView

@synthesize iconImage;

//_________________________________________________________________
- (id)initWithFrame:(CGRect)frame
{
    self = [super initWithFrame:frame];
    
    if (self) {
        // Initialization code
        self.opaque = NO;
    }
    
    return self;
}

//_________________________________________________________________
- (void)dealloc
{
   self.iconImage = nil;
   [super dealloc];
}

//_________________________________________________________________
- (void)setHintIcon:(NSString *)iconName hintText:(NSString *)text
{
   self.iconImage = [UIImage imageNamed:iconName];
   hintText = text;
}

//_________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextSetRGBFillColor(ctx, 0.3f, 0.3f, 0.3f, 0.7f);
   
   CGContextFillRect(ctx, rect);

   //Draw the hint's text.
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   [hintText drawInRect:CGRectMake(0.f, 350.f, rect.size.width, rect.size.height) withFont:[UIFont systemFontOfSize:32] lineBreakMode:UILineBreakModeWordWrap alignment:UITextAlignmentCenter];

   const CGPoint iconPlace = CGPointMake(rect.size.width / 2.f - 40.f, rect.size.height / 2.f - 40.f);
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, CGRectMake(iconPlace.x, iconPlace.y, 80.f, 80.f));
   [iconImage drawAtPoint:iconPlace];
}

//_________________________________________________________________
- (void) handleTap:(UITapGestureRecognizer *)tap
{
   self.hidden = YES;
}

@end
