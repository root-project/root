#import <UIKit/UIKit.h>

//View which contains hint's pictogram and textual description.

@interface HintView : UIView
- (void) setHintIcon : (NSString*) iconName hintText : (NSString*)text;
@end
