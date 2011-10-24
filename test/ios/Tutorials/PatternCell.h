//
//  PatternCell.h
//  Tutorials
//
//  Created by Timur Pocheptsov on 8/11/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>


@interface PatternCell : UIView {
   unsigned patternIndex;
   BOOL solid;
}

- (id) initWithFrame : (CGRect) frame andPattern : (unsigned) index;
- (void) dealloc;

- (void) setAsSolid;
- (void) drawRect : (CGRect) rect;

@end
