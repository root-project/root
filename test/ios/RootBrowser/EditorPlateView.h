#import <UIKit/UIKit.h>

@class EditorView;

@interface EditorPlateView : UIView {
@private
   UIImage *plateImage;
   NSString *editorName_;
   UILabel *editorLabel;
   
   UIImage *arrowImage;
   UIImageView *arrowImageView;
   
   EditorView *topView;
}

@property (nonatomic, retain) NSString *editorName;
@property (assign) UIImageView *arrowImageView;

+ (CGFloat) plateHeight;

- (id) initWithFrame : (CGRect)frame editorName : (NSString *) name topView : (EditorView *) tv;

- (void) handleTap : (UITapGestureRecognizer *) tap;

@end
