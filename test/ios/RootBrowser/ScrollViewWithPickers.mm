#import "ScrollViewWithPickers.h"


@implementation ScrollViewWithPickers

//____________________________________________________________________________________________________
- (UIView *) hitTest : (CGPoint)point withEvent : (UIEvent *)event
{
   UIView * v = [super hitTest : point withEvent : event];

   if ([v isKindOfClass : [UIPickerView class]] || [v.superview isKindOfClass : [UIPickerView class]]) {
      self.canCancelContentTouches = NO;
      self.delaysContentTouches = NO;
   } else {
      self.canCancelContentTouches = YES;
      self.delaysContentTouches = YES;
   }

   return v;
}


@end
