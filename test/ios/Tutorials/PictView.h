#import <UIKit/UIKit.h>

/*
   Small (50x50) view to draw a hint's pictogram.
*/

@interface PictView : UIImageView
- (id) initWithFrame : (CGRect)frame andIcon:(NSString *)iconName;
@end
