#import <UIKit/UIKit.h>

///////////////////////////////////////////////////////////
//  Custom view, subview for a detail view.
//  Delegates all graphics to C++ code.
//  This is view for pad in "editable state".
//  TODO: find better class name.
///////////////////////////////////////////////////////////

namespace ROOT {
namespace iOS {

class Pad;

}
}

@class ROOTObjectController;
@class SelectionView;

@interface PadView : UIView {
   ROOT::iOS::Pad *pad;
   
   ROOTObjectController *controller;
   
   CGFloat currentScale;

   BOOL panActive;
   
   SelectionView *selectionView;
   
   CGPoint tapPt;
   BOOL processSecondTap;
}

@property (nonatomic, assign) SelectionView *selectionView;

- (id) initWithFrame : (CGRect)frame controller : (ROOTObjectController *)c forPad : (ROOT::iOS::Pad*)pad;
- (void) dealloc;

- (void) setPad : (ROOT::iOS::Pad *)newPad;
- (void) drawRect : (CGRect)rect;
- (void) clearPad;

- (BOOL) pointOnSelectedObject : (CGPoint) pt;
- (void) addPanRecognizer;
- (void) removePanRecognizer;

- (void) handleSingleTap;
- (void) handleDoubleTap;

@end
