//
//This is scroll-view for pad in editable mode.
//
//The problem - we are processing too many gestures:
//scroll view itself supports several gestures like
//double taps, pans, pinches. Pad view which is inside
//this scroll view, also supports gestures like: tap (to select an object),
//pan (to zoom/unzoom plot's axes).

#import <UIKit/UIKit.h>

@interface PadEditorScrollView : UIScrollView
@end
