#import <MessageUI/MessageUI.h>
#import <UIKit/UIKit.h>

#import "IOSFileContainer.h"

@class ScrollViewWithPadView;
@class PadImageScrollView;
@class ObjectInspector;
@class ObjectShortcut;
@class EditorView;
@class PadView;

namespace ROOT {
namespace iOS {

//Pad to draw object.
class Pad;

}
}

namespace ROOT_IOSObjectController {

enum Mode {
   ocmNavigation,
   ocmEdit
};

}

class TObject;

@interface ROOTObjectController : UIViewController <UIScrollViewDelegate, MFMailComposeViewControllerDelegate> {
@private
   ROOT_IOSObjectController::Mode mode;

   EditorView *editorView;
   ObjectInspector *objectInspector;
   
   IBOutlet ScrollViewWithPadView *padScrollView;
   IBOutlet UIScrollView *navigationScrollView;
   
   PadView *editablePadView;

   ROOT::iOS::FileContainer *fileContainer;

   TObject *selectedObject;
   
   BOOL zoomed;
   
   PadImageScrollView *navScrolls[3];

   unsigned currentObject;
   unsigned nextObject;
   unsigned previousObject;
   
   UIBarButtonItem *editBtn;   
}

@property (nonatomic, retain) ScrollViewWithPadView *padScrollView;
@property (nonatomic, retain) UIScrollView *navigationScrollView;


- (void) setNavigationForObjectWithIndex : (unsigned) index fromContainer : (ROOT::iOS::FileContainer *)fileContainer;
- (void) handleDoubleTapOnPad : (CGPoint)tapPt;
- (void) objectWasSelected : (TObject *)object;
- (void) objectWasModifiedUpdateSelection : (BOOL)needUpdate;
- (void) setupObjectInspector;

- (ROOT::iOS::EHistogramErrorOption) getErrorOption;
- (void) setErrorOption : (ROOT::iOS::EHistogramErrorOption) errorOption;

- (BOOL) markerIsOn;
- (void) setMarker : (BOOL)on;

@end
