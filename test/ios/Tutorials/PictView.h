//
//  PictView.h
//  Tutorials
//
//  Created by Timur Pocheptsov on 7/17/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <UIKit/UIKit.h>

/*
   Small (50x50) view to draw a hint's pictogram.
*/

@interface PictView : UIView {
   UIImage *image;
}

- (id)initWithFrame:(CGRect)frame andIcon:(NSString *)iconName;
- (void) dealloc;

- (void) drawRect:(CGRect)rect;

@property (nonatomic, retain) UIImage *image;

@end
