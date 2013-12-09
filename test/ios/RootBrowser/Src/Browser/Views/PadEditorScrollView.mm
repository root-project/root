#import "PadEditorScrollView.h"
#import "PadView.h"

@implementation PadEditorScrollView

//____________________________________________________________________________________________________
- (UIView *) hitTest : (CGPoint) point withEvent : (UIEvent *) event
{  
   UIView * const v = [super hitTest : point withEvent : event];
   
   if ([v isKindOfClass : [PadView class]]) {
      PadView * const padView = (PadView *)v;

      if ([padView pointOnSelectedObject : [self convertPoint : point toView : padView]]) {
         //If we have some object in this point, we can probably pan (zoom/unzoom an axis)
         //or just tap on object, selecting it.
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
