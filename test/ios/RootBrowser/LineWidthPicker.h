#import <UIKit/UIKit.h>

@class LineWidthCell;

@interface LineWidthPicker : UIView {
@private
   LineWidthCell *lineWidthView;
   UIImage *backgroundImage;
}

- (void) setLineWidth : (float)width;

@end
