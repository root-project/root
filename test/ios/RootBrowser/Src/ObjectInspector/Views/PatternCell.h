#import <UIKit/UIKit.h>

@interface PatternCell : UIView

- (instancetype) initWithFrame : (CGRect) frame andPattern : (unsigned) index;
- (void) setAsSolid;

@end
