#import "ScrollViewWithPadView.h"
#import "PadView.h"

@implementation ScrollViewWithPadView

//____________________________________________________________________________________________________
- (UIView *) hitTest : (CGPoint)point withEvent : (UIEvent *)event
{
   UIView * v = [super hitTest : point withEvent : event];

   if ([v isKindOfClass : [PadView class]]) {
      PadView *padView = (PadView *)v;

      if ([padView pointOnSelectedObject : [self convertPoint : point toView : padView]]) {
         self.canCancelContentTouches = NO;
         self.delaysContentTouches = NO;
         [padView addPanRecognizer];
      } else {
         [padView removePanRecognizer];
         self.canCancelContentTouches = YES;
         self.delaysContentTouches = YES;
      }
   }

   return v;
}

@end
