#include <QuartzCore/QuartzCore.h>
#import <UIKit/UIKit.h>

@class NSTimer;

namespace ROOT {
namespace iOS {
namespace Demos {

class DemoBase;

}

class Pad;
}
}

@class PadOptionsController;
@class SelectionView;
@class PictView;
@class HintView;
@class PadView;

class FontManagerWrapper;
class PainterWrapper;
class PadWrapper;

enum ETutorialsMode {
   kTAZoom,
   kTASelect
};

enum ETutorialsDefaults {
   kTDNOfPads = 2
};

@interface DetailViewController : UIViewController <UIPopoverControllerDelegate, UISplitViewControllerDelegate, UIScrollViewDelegate> {
   NSTimer *animationTimer;
   unsigned currentFrame;

   ROOT::iOS::Pad *pad;

   //Depending on more, either parentView of
   //scrollViews is/are parent(s) of padViews.
   UIView *parentView;
   UIScrollView *scrollViews[kTDNOfPads];
   PadView *padViews[kTDNOfPads];

   //Transparent view with selected object.
   SelectionView *selectionViews[kTDNOfPads];

   UIPanGestureRecognizer *padPanGestures[kTDNOfPads];
   UITapGestureRecognizer *padTapGestures[kTDNOfPads];
   
   unsigned activeView;
   
   CGSize oldSizes;

   ROOT::iOS::Demos::DemoBase *activeDemo;

   //Transparent view with a text
   //and a pictogram for a hint.
   HintView *hintView;
  
   //Small views with pictograms: hints. 
   PictView *panPic;
   PictView *pinchPic;
   PictView *doubleTapPic;
   PictView *rotatePic;
   PictView *singleTapPic;
   //PictView *emptyPic;
   
   //Text for hints.
   NSString *pinchHintText;
   NSString *panHintText;
   NSString *doubleTapHintText;
   NSString *rotateHintText;
   NSString *singleTapHintText;
   //NSString *emptyHintText;
   
   //Either zoom or selection.
   ETutorialsMode appMode;
   
   IBOutlet UITabBar *tb;
//   IBOutlet UIView *editor;



   IBOutlet UIView *help;
   
   PadOptionsController *padController_;
   
   UIPopoverController *editorPopover_;
   
   BOOL activeAnimation;
}


@property (nonatomic, retain) IBOutlet UIToolbar *toolbar;
@property (nonatomic, retain) id detailItem;
@property (nonatomic, retain) IBOutlet UILabel *detailDescriptionLabel;
@property (nonatomic, retain) IBOutlet UIView *help;
@property (nonatomic, retain) PadOptionsController * padController;
@property (nonatomic, retain) UIPopoverController *editorPopover;

- (void) setActiveDemo : (ROOT::iOS::Demos::DemoBase *)demo;
- (void) onTimer;

- (void)animationDidStop:(CAAnimation *)theAnimation finished:(BOOL)flag;

- (IBAction) zoomButtonPressed;
- (IBAction) editButtonPressed : (id) sender;
- (IBAction) selectButtonPressed;

- (void) showPanHint;
- (void) showPinchHint;
- (void) showDoubleTapHint;
- (void) showRotationHint;
- (void) showEmptyHint;
- (void) showSingleTapHint;

//- (void) handleDoubleTap : (UITapGestureRecognizer *)tap;
- (void) handleDoubleTapPad : (UITapGestureRecognizer *)tap;
- (IBAction) showHelp;


@end
