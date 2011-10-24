#import <UIKit/UIKit.h>

@class ScrollViewWithPickers;
@class EditorPlateView;

//Hoho! As soon as I use Objective-C++, I can use namespaces! "Yeaaahhh, that's good!" (c) Duke Nukem.

namespace ROOT_IOSObjectInspector {

enum {
   evMaxComponents = 5,
   evMaxStates = 1 << evMaxComponents
};

}

@interface EditorView : UIView {
@private
   UILabel *editorTitle;

   ScrollViewWithPickers *scrollView;

   CGFloat plateYs[ROOT_IOSObjectInspector::evMaxStates * ROOT_IOSObjectInspector::evMaxComponents];
   CGFloat viewYs[ROOT_IOSObjectInspector::evMaxStates * ROOT_IOSObjectInspector::evMaxComponents];
   
   UIView *plates[ROOT_IOSObjectInspector::evMaxComponents];
   UIView *views[ROOT_IOSObjectInspector::evMaxComponents];
   UIView *containers[ROOT_IOSObjectInspector::evMaxComponents];   

   unsigned nStates;
   unsigned nEditors;
   unsigned currentState;
   
   int newOpened;
   
   BOOL animation;
}

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
