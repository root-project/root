#import "RangeSlider.h"

@interface RangeSlider (PrivateMethods)
-(float) xForValue : (float)value;
-(float) valueForX : (float)x;
-(void)updateTrackHighlight;
@end

@implementation RangeSlider

@synthesize minimumValue, maximumValue, minimumRange, selectedMinimumValue, selectedMaximumValue;

//____________________________________________________________________________________________________
- (id) initWithFrame : (CGRect)frame
{
   self = [super initWithFrame : frame];

   if (self) {
      // Set the initial state
      minimumValue = 0.f;
      maximumValue = 1.f;
      selectedMinimumValue = 0.f;
      selectedMaximumValue = 1.f;

      minThumbOn = NO;
      maxThumbOn = NO;

      minimumRange = 4.f;//FIXME
      padding = 20;

      UIImageView *trackBackground = [[UIImageView alloc] initWithImage : [UIImage imageNamed : @"bar-background.png"]];
      trackBackground.frame = CGRectMake(0.f, self.frame.size.height / 2 - trackBackground.frame.size.height / 2, self.frame.size.width - padding * 2, trackBackground.frame.size.height);
      trackBackground.center = CGPointMake(self.frame.size.width / 2, self.frame.size.height / 2);
      [self addSubview : trackBackground];
        
      track = [[UIImageView alloc] initWithImage : [UIImage imageNamed:@"bar-highlight.png"]];
      track.frame = CGRectMake(0.f, self.frame.size.height / 2 - track.frame.size.height / 2, self.frame.size.width - padding * 2, track.frame.size.height);
      track.center = CGPointMake(self.frame.size.width / 2, self.frame.size.height / 2);
      [self addSubview : track];
        
      minThumb = [[UIImageView alloc] initWithImage : [UIImage imageNamed : @"handle.png"] highlightedImage : [UIImage imageNamed : @"handle-hover.png"]];
      minThumb.frame = CGRectMake(0.f, 0.f, self.frame.size.height,self.frame.size.height);
      minThumb.contentMode = UIViewContentModeCenter;
		minThumb.center = CGPointMake([self xForValue : selectedMinimumValue], self.frame.size.height / 2);
		[self addSubview : minThumb];
        
      maxThumb = [[UIImageView alloc] initWithImage : [UIImage imageNamed : @"handle.png"] highlightedImage : [UIImage imageNamed : @"handle-hover.png"]];
      maxThumb.frame = CGRectMake(0.f, 0.f, self.frame.size.height, self.frame.size.height);
      maxThumb.contentMode = UIViewContentModeCenter;
		maxThumb.center = CGPointMake([self xForValue : selectedMaximumValue], self.frame.size.height / 2);
      [self addSubview : maxThumb];

      [self updateTrackHighlight];
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) setSliderMin : (float)min max : (float)max selectedMin : (float)sMin selectedMax : (float)sMax
{
   minimumValue = min;
   maximumValue = max;
   selectedMinimumValue = sMin;
   selectedMaximumValue = sMax;
   
   minThumb.center = CGPointMake([self xForValue : selectedMinimumValue], self.frame.size.height / 2);
   maxThumb.center = CGPointMake([self xForValue : selectedMaximumValue], self.frame.size.height / 2);
   
   [self updateTrackHighlight];
}


//____________________________________________________________________________________________________
- (CGFloat) getMinThumbX
{
   return minThumb.center.x;
}

//____________________________________________________________________________________________________
- (CGFloat) getMaxThumbX
{
   return maxThumb.center.x;
}

//____________________________________________________________________________________________________
-(float) xForValue : (float)value
{
   return (self.frame.size.width-(padding * 2))*((value - minimumValue) / (maximumValue - minimumValue)) + padding;
}

//____________________________________________________________________________________________________
-(float) valueForX : (float)x
{
   return minimumValue + (x - padding) / (self.frame.size.width - (padding * 2)) * (maximumValue - minimumValue);
}

//____________________________________________________________________________________________________
-(BOOL) continueTrackingWithTouch : (UITouch *)touch withEvent : (UIEvent *)event
{
   if(!minThumbOn && !maxThumbOn)
      return NO;
    
   CGPoint touchPoint = [touch locationInView:self];
   if(minThumbOn) {
      minThumb.center = CGPointMake(MAX([self xForValue : minimumValue], MIN(touchPoint.x, [self xForValue : selectedMaximumValue - minimumRange])), minThumb.center.y);
      selectedMinimumValue = [self valueForX : minThumb.center.x];     
   }
    
   if (maxThumbOn) {
      maxThumb.center = CGPointMake(MIN([self xForValue : maximumValue], MAX(touchPoint.x, [self xForValue : selectedMinimumValue + minimumRange])), maxThumb.center.y);
      selectedMaximumValue = [self valueForX : maxThumb.center.x];
   }
   
   [self updateTrackHighlight];
   [self setNeedsDisplay];
    
   [self sendActionsForControlEvents : UIControlEventValueChanged];

   return YES;
}

//____________________________________________________________________________________________________
-(BOOL) beginTrackingWithTouch : (UITouch *)touch withEvent : (UIEvent *)event
{
   CGPoint touchPoint = [touch locationInView : self];
   if (CGRectContainsPoint(minThumb.frame, touchPoint)) {
      minThumbOn = YES;
   } else if (CGRectContainsPoint(maxThumb.frame, touchPoint)) {
      maxThumbOn = YES;
   }

   return YES;
}

//____________________________________________________________________________________________________
-(void) endTrackingWithTouch : (UITouch *)touch withEvent : (UIEvent *)event
{
   minThumbOn = false;
   maxThumbOn = false;
}

//____________________________________________________________________________________________________
-(void) updateTrackHighlight
{
   track.frame = CGRectMake(minThumb.center.x, track.center.y - (track.frame.size.height / 2),
                            maxThumb.center.x - minThumb.center.x, track.frame.size.height);
}

@end
