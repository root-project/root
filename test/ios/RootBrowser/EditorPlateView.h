#import <UIKit/UIKit.h>

@class EditorView;

@interface EditorPlateView : UIView

@property (nonatomic, retain) NSString *editorName;
@property (nonatomic, retain) UIImageView *arrowImageView;

+ (CGFloat) plateHeight;
- (id) initWithFrame : (CGRect)frame editorName : (NSString *) name topView : (EditorView *) tv;

@end
