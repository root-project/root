#import <MessageUI/MessageUI.h>
#import <UIKit/UIKit.h>

#import "FileUtils.h"

@class ScrollViewWithPadView;

class TObject;

@interface ObjectViewController : UIViewController <UIScrollViewDelegate, MFMailComposeViewControllerDelegate>

- (void) setNavigationForObjectWithIndex : (unsigned) index fromContainer : (ROOT::iOS::Browser::FileContainer *) fileContainer;
- (void) handleDoubleTapOnPad : (CGPoint) tapPt;
- (void) objectWasSelected : (TObject *) object;
- (void) objectWasModifiedUpdateSelection : (BOOL) needUpdate;
- (void) setupObjectInspector;

- (ROOT::iOS::Browser::EHistogramErrorOption) getErrorOption;
- (void) setErrorOption : (ROOT::iOS::Browser::EHistogramErrorOption) errorOption;

- (BOOL) markerIsOn;
- (void) setMarker : (BOOL)on;

@end
