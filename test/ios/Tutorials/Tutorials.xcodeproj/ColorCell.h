#import <UIKit/UIKit.h>

//
//This is a small and simple custom view, used inside color picker (UIPickerView).
//

@interface ColorCell : UIView

- (void) setRGB : (const double *) rgb;

@end
