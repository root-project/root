#import <UIKit/UIKit.h>


//
//EditorView is a small panel to the right of
//pad, contains widgets to modify pad's contents.
//To make it possible to include more staff,
//every sub-editor (like line attributes editor,
//or filled area editor) is placed into
//the collapsing panel.
//

@class EditorPlateView;

@interface EditorView : UIView 

+ (CGFloat) editorAlpha;
+ (CGFloat) editorWidth;
+ (CGFloat) editorHeight;
+ (CGFloat) scrollWidth;
+ (CGFloat) scrollHeight;

- (void) removeAllEditors;
- (void) propertyUpdated;
- (void) addSubEditor : (UIView *)element withName : (NSString *)editorName;
- (void) correctFrames;
- (void) plateTapped : (EditorPlateView *) plate;
- (void) setEditorTitle : (const char*)title;

@end
