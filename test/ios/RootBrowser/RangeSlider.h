//Very nice double range slider
//by Mal Curtis (http://buildmobile.com).
//Tpochep had to modify it to make it work: initWithFrame was
//wrong (coordinates for UIImageView's),
//added setSliderxxxx method.

#import <UIKit/UIKit.h>

@interface RangeSlider : UIControl {
@private
   float minimumValue;
   float maximumValue;
   float minimumRange;
   float selectedMinimumValue;
   float selectedMaximumValue;

   float padding;
    
   BOOL maxThumbOn;
   BOOL minThumbOn;
    
   UIImageView *minThumb;
   UIImageView *maxThumb;
   UIImageView *track;
}

@property(nonatomic) float minimumValue;
@property(nonatomic) float maximumValue;
@property(nonatomic) float minimumRange;
@property(nonatomic) float selectedMinimumValue;
@property(nonatomic) float selectedMaximumValue;

- (void) setSliderMin : (float)min max : (float)max selectedMin : (float)sMin selectedMax : (float)sMax;

- (CGFloat) getMinThumbX;
- (CGFloat) getMaxThumbX;

@end
