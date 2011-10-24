//
//  Shortcuts.m
//  root_browser
//
//  Created by Timur Pocheptsov on 8/22/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import "Shortcuts.h"

@implementation ShorcutUtil

+(void) placeShortcuts :(NSMutableArray *)shortcuts inScrollView : (UIScrollView *) scrollView withSize : (CGSize) size andSpace : (CGFloat) addSpace
{
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

@end
