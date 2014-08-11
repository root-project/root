#import <UIKit/UIKit.h>

///////////////////////////////////////////////////////////
//  Custom view, subview for a detail view.
//  Delegates all graphics to C++ code.
///////////////////////////////////////////////////////////

@class SelectionView;

namespace ROOT {
namespace iOS {

class Pad;

}
}

@interface PadView : UIView

- (id) initWithFrame : (CGRect)frame forPad : (ROOT::iOS::Pad*)pad;

- (void) clearPad;
- (void) setSelectionView : (SelectionView *) sv;
- (void) setProcessPan : (BOOL)p;
- (void) setProcessTap : (BOOL)t;


@end
