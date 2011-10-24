#import <UIKit/UIKit.h>


@interface HintView : UIView {
   UIImage *iconImage;
   NSString *hintText;
}

- (id) initWithFrame:(CGRect)frame;
- (void) dealloc;

- (void) setHintIcon: (NSString*) iconName hintText:(NSString*)text;
- (void) drawRect:(CGRect)rect;
- (void) handleTap : (UITapGestureRecognizer *)tap;

@property (nonatomic, retain) UIImage * iconImage;

@end
