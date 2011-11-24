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

@interface PadView : UIView

@property (nonatomic, retain) SelectionView *selectionView;

- (id) initWithFrame : (CGRect)frame controller : (ROOTObjectController *)c forPad : (ROOT::iOS::Pad*)pad;


- (void) setPad : (ROOT::iOS::Pad *)newPad;
- (void) clearPad;

- (BOOL) pointOnSelectedObject : (CGPoint) pt;
- (void) addPanRecognizer;
- (void) removePanRecognizer;

@end
