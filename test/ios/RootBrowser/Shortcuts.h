//
//  Shortcuts.h
//  root_browser
//
//  Created by Timur Pocheptsov on 8/22/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>


@interface ShorcutUtil : NSObject {
}

+(void) placeShortcuts :(NSMutableArray *)shortcuts inScrollView : (UIScrollView *) scrollView withSize : (CGSize) size andSpace : (CGFloat) space;

@end


