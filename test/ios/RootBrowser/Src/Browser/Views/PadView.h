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

@class ObjectViewController;
@class PadSelectionView;

@interface PadView : UIView

@property (nonatomic, readonly) PadSelectionView *selectionView;
@property (nonatomic) BOOL zoomed;

- (instancetype) initWithFrame : (CGRect) frame controller : (ObjectViewController *) c forPad : (ROOT::iOS::Pad*) pad;
- (instancetype) initImmutableViewWithFrame : (CGRect) frame;

- (void) setPad : (ROOT::iOS::Pad *) newPad;
- (void) clearPad;

- (BOOL) pointOnSelectedObject : (CGPoint) pt;
- (void) addPanRecognizer;
- (void) removePanRecognizer;

@end
