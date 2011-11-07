#import <UIKit/UIKit.h>

@interface PatternCell : UIView

- (id) initWithFrame : (CGRect) frame andPattern : (unsigned) index;
- (void) setAsSolid;

@end
