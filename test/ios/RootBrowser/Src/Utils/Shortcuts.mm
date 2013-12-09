#import <cassert>

#import "Shortcuts.h"

namespace ROOT {
namespace iOS {
namespace Browser {

void PlaceShortcutsInScrollView(NSMutableArray *shortcuts, UIScrollView *scrollView, const CGSize &size, CGFloat addSpace)
{
   //We assume, that 'shortcuts' (views) are already subviews of a scrollview, I do not check it.

   assert(shortcuts != nil && "PlaceShortcutsInScrollView, parameter 'shortcuts' is nil");
   assert(scrollView != nil && "PlaceShortcutsInScrollView, parameter 'scrollView' is nil");
   
   const CGRect scrollFrame = scrollView.frame;
   const CGFloat shortcutWidth = size.width;
   const CGFloat shortcutHeight = size.height;
   const unsigned nPicksInRow = scrollFrame.size.width / (shortcutWidth + addSpace);
   const CGFloat addXY = (scrollFrame.size.width - (shortcutWidth + addSpace) * nPicksInRow) / 2;
   
   NSEnumerator *enumerator = [shortcuts objectEnumerator];
   UIView *v = [enumerator nextObject];
   for (unsigned n = 0; v; v = [enumerator nextObject], ++n) {
      const unsigned col = n % nPicksInRow;
      const unsigned row = n / nPicksInRow;
      
      const CGFloat x = addXY + addSpace / 2 + col * (shortcutWidth + addSpace);
      const CGFloat y = row * shortcutHeight + addXY;

      CGRect frame = v.frame;
      frame.origin = CGPointMake(x, y);
      v.frame = frame;
   }
   
   scrollView.contentSize = CGSizeMake(scrollFrame.size.width, addXY + ([shortcuts count] + nPicksInRow - 1) / nPicksInRow * shortcutHeight);
   scrollView.contentOffset = CGPointZero;
}

}
}
}
