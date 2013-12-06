#import <CoreGraphics/CoreGraphics.h>

#import "HintView.h"

@implementation HintView {
   UIImage *iconImage;
   NSString *hintText;
}

//_________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   self = [super initWithFrame : frame];
    
   if (self) {
      // Initialization code
      self.opaque = NO;
   }
    
   return self;
}

//_________________________________________________________________
- (void) setHintIcon : (NSString *)iconName hintText : (NSString *)text
{
   iconImage = [UIImage imageNamed : iconName];
   hintText = text;
}

//_________________________________________________________________
- (void) drawRect : (CGRect)rect
{
   CGContextRef ctx = UIGraphicsGetCurrentContext();
   CGContextSetRGBFillColor(ctx, 0.3f, 0.3f, 0.3f, 0.7f);
   
   CGContextFillRect(ctx, rect);

   const CGRect textRect = CGRectMake(0.f, 350.f, rect.size.width, rect.size.height);
   UIFont * const font = [UIFont systemFontOfSize : 32.f];
   NSMutableParagraphStyle * const paragraphStyle = [[NSParagraphStyle defaultParagraphStyle] mutableCopy];
   paragraphStyle.lineBreakMode = NSLineBreakByWordWrapping;
   paragraphStyle.alignment = NSTextAlignmentCenter;
   NSDictionary * const attributes = @{NSFontAttributeName : font, NSParagraphStyleAttributeName : paragraphStyle,
                                       NSForegroundColorAttributeName : [UIColor whiteColor]};
   
   [hintText drawInRect : textRect withAttributes : attributes];

   const CGPoint iconPlace = CGPointMake(rect.size.width / 2.f - 40.f, rect.size.height / 2.f - 40.f);
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillRect(ctx, CGRectMake(iconPlace.x, iconPlace.y, 80.f, 80.f));
   [iconImage drawAtPoint : iconPlace];
}

//_________________________________________________________________
- (void) handleTap : (UITapGestureRecognizer *) tap
{
   self.hidden = YES;
}

@end
