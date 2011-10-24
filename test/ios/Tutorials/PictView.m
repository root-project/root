//
//  PictView.m
//  Tutorials
//
//  Created by Timur Pocheptsov on 7/17/11.
//  Copyright 2011 __MyCompanyName__. All rights reserved.
//

#import <QuartzCore/QuartzCore.h>

#import "PictView.h"

@implementation PictView

@synthesize image;

//_________________________________________________________________
- (id) initWithFrame:(CGRect)frame andIcon:(NSString *)iconName
{
   self = [super initWithFrame:frame];
   
   if (self) {
      // Initialization code
      self.image = [UIImage imageNamed:iconName];

      //View is transparent with shadow (under non-transparent pixels in a picture).
      self.opaque = NO;
      self.layer.shadowColor = [UIColor blackColor].CGColor;
      self.layer.shadowOpacity = 0.7f;
      self.layer.shadowOffset = CGSizeMake(3.f, 3.f);
   }
   
   return self;
}

//_________________________________________________________________
- (void)dealloc
{
   self.image = nil;
   [super dealloc];
}

//_________________________________________________________________
- (void)drawRect:(CGRect)rect
{
   [image drawInRect:rect blendMode:kCGBlendModeOverlay alpha:0.3f];
}

@end
