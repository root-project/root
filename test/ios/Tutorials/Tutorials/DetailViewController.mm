#import <QuartzCore/QuartzCore.h>

#import "PadOptionsController.h"
#import "DetailViewController.h"
#import "RootViewController.h"
#import "SelectionView.h"
#import "HintView.h"
#import "PictView.h"
#import "PadView.h"

//C++ code.
//ROOT
#import "IOSPad.h"

//Demos.
#import "DemoBase.h"

/*
DetailViewController has several views - two pad views (to make animation while changing demos),
two scroll views to be parent of pad-views (to enable zoom and scroll)
and parent view. PadView can be placed in a parent view, or in a scroll-view (which will be placed
in parent view). Actually, I do not need all these view, but there's no serious reason to
modify this code already. This was a testbed.
*/

namespace {

enum ETutorialsMode {
   kTAZoom,
   kTASelect
};

enum ETutorialsDefaults {
   kTDNOfPads = 2
};

}

//Hidden implementation details.
@interface DetailViewController () {

   //"Editor"
   PadOptionsController * padController;
   UIPopoverController *editorPopover;

   //Transparent view with a text
   //and a pictogram for a hint.
   HintView *hintView;
  
   //Small views with pictograms: hints. 
   PictView *panPic;
   PictView *pinchPic;
   PictView *doubleTapPic;
   PictView *rotatePic;
   PictView *singleTapPic;

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
   
   //Either zoom or selection.
   ETutorialsMode appMode;
   
   BOOL activeAnimation;
}

@property (nonatomic, retain) UIPopoverController *popoverController;

- (void) showPanHint;
- (void) showPinchHint;
- (void) showDoubleTapHint;
- (void) showRotationHint;
- (void) showSingleTapHint;
- (void) handleDoubleTapPad : (UITapGestureRecognizer *)tap;

@end



@implementation DetailViewController

//These are generated declarations.
@synthesize toolbar;
@synthesize popoverController;

//This was "generated" by me.
@synthesize help;
@synthesize tabBar;

#pragma mark - Managing the detail item

//_________________________________________________________________
- (void) initCPPObjects
{
   pad = new ROOT::iOS::Pad(640, 640);
}

//_________________________________________________________________
- (void) initMainViews
{
   const CGPoint padCenter = self.view.center;
   CGRect padRect = CGRectMake(padCenter.x - 320.f, padCenter.y - 310.f, 640.f, 640.f);
   
   oldSizes.width = 640.f;
   oldSizes.height = 640.f;
   
   parentView = [[UIView alloc] initWithFrame:padRect];
   parentView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;

   [self.view addSubview:parentView];

   //Trick with shadow and shadow path: 
   //http://nachbaur.com/blog/fun-shadow-effects-using-custom-calayer-shadowpaths
   //Many thanks.
   parentView.layer.shadowColor = [UIColor blackColor].CGColor;
   parentView.layer.shadowOpacity = 0.7f;
   parentView.layer.shadowOffset = CGSizeMake(10.f, 10.f);
   UIBezierPath *path = [UIBezierPath bezierPathWithRect:parentView.bounds];
   parentView.layer.shadowPath = path.CGPath;
   ///
   padRect.origin.x = 0.f, padRect.origin.y = 0.f;
   for (unsigned i = 0; i < 2; ++i) {// < kTDNOfPads
      scrollViews[i] = [[UIScrollView alloc] initWithFrame:padRect];
      scrollViews[i].backgroundColor = [UIColor darkGrayColor];
      scrollViews[i].delegate = self;
      padViews[i] = [[PadView alloc] initWithFrame : padRect forPad : pad];
      scrollViews[i].contentSize = padViews[i].frame.size;
      [scrollViews[i] addSubview:padViews[i]];
      //
      scrollViews[i].minimumZoomScale = 1.f;
      scrollViews[i].maximumZoomScale = 1280.f / 640.f;
      [scrollViews[i] setZoomScale:1.f];
      [parentView addSubview:scrollViews[i]];
   }

   parentView.hidden = YES;
   //   
   activeView = 0;
   
   padRect = CGRectMake(padCenter.x - 320.f, padCenter.y - 310.f, 640.f, 640.f);
   
   for (unsigned i = 0; i < 2; ++i) { // < kTDNOfPads
      selectionViews[i] = [[SelectionView alloc] initWithFrame:padRect];
      selectionViews[i].autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
      [self.view addSubview:selectionViews[i]];
      selectionViews[i].hidden = YES;   
      selectionViews[i].opaque = NO;
   }
}

//_________________________________________________________________
- (void) initHints
{
   //"Hint" is a semi-transparent view, which shows gesture icon and some textual description.
   //Pictogramms.
   CGRect pictRect = CGRectMake(670.f, 450.f, 50.f, 50.f);
   pinchPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"pinch_gesture_icon_small.png"];
   pinchPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:pinchPic];
   UITapGestureRecognizer *pinchTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showPinchHint)];
   [pinchPic addGestureRecognizer:pinchTap];
   pinchPic.hidden = YES;

   pictRect.origin.y = 520;
   panPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"pan_gesture_icon_small.png"];
   panPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:panPic];
   UITapGestureRecognizer *panTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showPanHint)];
   [panPic addGestureRecognizer:panTap];
   panPic.hidden = YES;
   
   pictRect.origin.y = 590;
   doubleTapPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"double_tap_gesture_icon_small.png"];
   doubleTapPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:doubleTapPic];
   UITapGestureRecognizer *dtapTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showDoubleTapHint)];
   [doubleTapPic addGestureRecognizer:dtapTap];
   doubleTapPic.hidden = YES;

   rotatePic = [[PictView alloc] initWithFrame:pictRect andIcon:@"rotate_icon_small.png"];
   rotatePic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:rotatePic];
   UITapGestureRecognizer *rotTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showRotationHint)];
   [rotatePic addGestureRecognizer:rotTap];
   rotatePic.hidden = YES;
   
   singleTapPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"single_tap_icon_small.png"];
   singleTapPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:singleTapPic];
   UITapGestureRecognizer *singleTapTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showSingleTapHint)];
   [singleTapPic addGestureRecognizer:singleTapTap];
   singleTapPic.hidden = YES;

   const CGPoint center = self.view.center;
   CGRect rect = CGRectMake(center.x - 300.f, center.y - 290.f, 600.f, 600.f);
   hintView = [[HintView alloc] initWithFrame:rect];
   hintView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:hintView];
   UITapGestureRecognizer *hintTap = [[UITapGestureRecognizer alloc] initWithTarget:hintView action:@selector(handleTap:)];
   [hintView addGestureRecognizer:hintTap];
   hintView.hidden = YES;
}

//_________________________________________________________________
- (void)viewWillAppear:(BOOL)animated
{
   [super viewWillAppear:animated];
}

//_________________________________________________________________
- (void)viewDidAppear:(BOOL)animated
{
   [super viewDidAppear:animated];
}

//_________________________________________________________________
- (void)viewWillDisappear:(BOOL)animated
{
	[super viewWillDisappear:animated];
}

//_________________________________________________________________
- (void)viewDidDisappear:(BOOL)animated
{  
	[super viewDidDisappear:animated];
}

//_________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    return YES;
}

#pragma mark - Split view support

//_________________________________________________________________
- (void)splitViewController:(UISplitViewController *)svc willHideViewController:(UIViewController *)aViewController withBarButtonItem:(UIBarButtonItem *)barButtonItem forPopoverController: (UIPopoverController *)pc
{
   barButtonItem.title = @"Tutorials";
   NSMutableArray *items = [[self.toolbar items] mutableCopy];
   [items insertObject:barButtonItem atIndex:0];
   [self.toolbar setItems:items animated:YES];
   self.popoverController = pc;
}

//_________________________________________________________________
- (void)splitViewController:(UISplitViewController *)svc willShowViewController:(UIViewController *)aViewController invalidatingBarButtonItem:(UIBarButtonItem *)barButtonItem
{
   // Called when the view is shown again in the split view, invalidating the button and popover controller.
   NSMutableArray *items = [[self.toolbar items] mutableCopy];
   [items removeObjectAtIndex:0];
   [self.toolbar setItems:items animated:YES];
   self.popoverController = nil;
}

//_________________________________________________________________
- (void)viewDidLoad
{
   self.view.backgroundColor = [UIColor lightGrayColor];
   
   [self initCPPObjects];
   [self initMainViews];
   [self initHints];
   
   appMode = kTAZoom;
  
   UITapGestureRecognizer *tapGesture = [[UITapGestureRecognizer alloc] initWithTarget:self action:@selector(handleDoubleTapPad:)];
   [parentView addGestureRecognizer:tapGesture];
   tapGesture.numberOfTapsRequired = 2;
      
   //Load a help view from a nib file.
   [[NSBundle mainBundle] loadNibNamed:@"HelpView" owner:self options:nil];
   CGRect helpFrame = help.frame;
   helpFrame.origin.x = self.view.center.x - helpFrame.size.width / 2;
   helpFrame.origin.y = self.view.center.y - helpFrame.size.height / 2;
   help.frame = helpFrame;
   help.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:help];
         
   //Editor view and help view (created in a IB) are on top.
   [self.view bringSubviewToFront:help];
   
   //Shadow for editor.   
   tabBar.selectedItem = [tabBar.items objectAtIndex : 0];

   [super viewDidLoad];
}

//_________________________________________________________________
- (void)viewDidUnload
{
   [super viewDidUnload];

	// Release any retained subviews of the main view.
	// e.g. self.myOutlet = nil;
	self.popoverController = nil;
}

#pragma mark - Memory management

//_________________________________________________________________
- (void)didReceiveMemoryWarning
{
	// Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
	
	// Release any cached data, images, etc that aren't in use.
}

//_________________________________________________________________
- (void)dealloc
{
   //
   delete pad;
   //
}

//_________________________________________________________________
- (void) prepareHints
{
   if (appMode == kTAZoom) {
      panPic.hidden = NO;
      pinchPic.hidden = NO;
      doubleTapPic.hidden = NO;
      
      singleTapPic.hidden = YES;
      rotatePic.hidden = YES;
//      emptyPic.hidden = YES;
      //Hide selection pictograms.
   } else {
      //Show selection or rotate pictogram.
      
      //Hide zoom mode's pictograms.
      panPic.hidden = YES;
      pinchPic.hidden = YES;
      doubleTapPic.hidden = YES;

      rotatePic.hidden = !activeDemo->Supports3DRotation();
      singleTapPic.hidden = !rotatePic.hidden;
   }
}

//_________________________________________________________________
- (void) setActiveDemo:(ROOT::iOS::Demos::DemoBase *)demo
{
   help.hidden = YES;
   
   if (demo != activeDemo) {
   
      selectionViews[0].hidden = YES;
      selectionViews[1].hidden = YES;
   
      parentView.hidden = NO;
      //Stop any animated demo (previously active).
      if (animationTimer) {
         [animationTimer invalidate];
         animationTimer = 0;
      }
      currentFrame = 0;
      //
      
      //Prepare to make an animation: remove one view, show another.
      activeDemo = demo;
      [self prepareHints];
   
      UIView * showView = 0;
      UIView * hideView = 0;
      
      const unsigned hide = activeView;
      const unsigned show = !hide;
      activeView = show;
   
      if (appMode == kTAZoom) {
         showView = scrollViews[show];
         hideView = scrollViews[hide];
      } else {
         showView = padViews[show];
         hideView = padViews[hide];         
      }

      //This is temporary hack.
      [padViews[activeView] setProcessPan:activeDemo->Supports3DRotation()];
      [padViews[activeView] setProcessTap:!activeDemo->Supports3DRotation()];

      //Remove old contents of pad, 
      //set pad's parameters (if required by demo)
      //reset demo (if required), add demo's primitives to pad.
      pad->Clear();
      activeDemo->AdjustPad(pad);
      activeDemo->ResetDemo();
      activeDemo->PresentDemo();
      
      //Repaint active view's content.
      [padViews[activeView] setNeedsDisplay];
      
      if (activeDemo->IsAnimated()) {
         //Start timer for animated demo.
         activeDemo->StartAnimation();
         animationTimer = [NSTimer scheduledTimerWithTimeInterval:0.5 / 25 target:self selector:@selector(onTimer) userInfo:nil repeats:YES];
      }
      
      //Make an animation: hide one view (demo), show another one.
      showView.hidden = NO;
      hideView.hidden = YES;
      // First create a CATransition object to describe the transition
      CATransition *transition = [CATransition animation];
      // Animate over 3/4 of a second
      transition.duration = 0.75;
      // using the ease in/out timing function
      transition.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
      // Now to set the type of transition.
      transition.type = kCATransitionReveal;
		transition.subtype = kCATransitionFromLeft;
      // Finally, to avoid overlapping transitions we assign ourselves as the delegate for the animation and wait for the
      // -animationDidStop:finished: message. When it comes in, we will flag that we are no longer transitioning.
      //transitioning = YES;
      transition.delegate = self;
      // Next add it to the containerView's layer. This will perform the transition based on how we change its contents.
      [parentView.layer addAnimation : transition forKey:nil];
   }
}

//_________________________________________________________________
- (void) dismissPopover
{
   [popoverController dismissPopoverAnimated : YES];
}

//_________________________________________________________________
- (void) onTimer
{
   if (currentFrame == activeDemo->NumOfFrames()) {
      [animationTimer invalidate];
      animationTimer = 0;
   } else {
      ++currentFrame;
      activeDemo->NextStep();      
      [padViews[activeView] setNeedsDisplay];
   }
}

//_________________________________________________________________
- (void) resizePadView:(unsigned)view
{
   UIScrollView *scroll = (UIScrollView *)padViews[view].superview;
   CGRect oldRect = padViews[view].frame;
   
   if (abs(640.f - oldRect.size.width) < 0.01 && (abs(640.f - oldRect.size.height) < 0.01))
      return;
      
   CGRect padRect = CGRectMake(0.f, 0.f, 640.f, 640.f);
   [padViews[view] removeFromSuperview];
   padViews[view] = [[PadView alloc] initWithFrame : padRect forPad : pad];
   [scroll addSubview:padViews[view]];
  
   scroll.minimumZoomScale = 1.f;
   scroll.maximumZoomScale = 2.f;
   [scroll setZoomScale:1.f];
   scroll.contentSize = padRect.size;
   scroll.contentOffset = CGPointMake(0.f, 0.f);
      
   oldSizes.width = 640.f;
   oldSizes.height = 640.f;
}

//_________________________________________________________________
-(void)animationDidStop:(CAAnimation *)theAnimation finished:(BOOL)flag
{
   //After one view was hidden, resize it's scale to 1 and
   //view itself to original size.
   if (appMode == kTAZoom) {
      unsigned inactiveView = activeView ? 0 : 1;
      [self resizePadView : inactiveView];
   }
}

#pragma mark - That's what I call action :)

//_________________________________________________________________
- (IBAction)zoomButtonPressed
{
   if (appMode == kTAZoom)
      return;
 
   //Zoom mode was selected.
   appMode = kTAZoom;
   //The mode was kTASelect previously.
   //Reparent pad views, now scrollview is a parent for a pad view.
   for (unsigned i = 0; i < 2; ++i) { // < kTDNOfPads.
      selectionViews[i].hidden = YES;
      [padViews[i] removeGestureRecognizer : padPanGestures[i]];
      [padViews[i] removeGestureRecognizer : padTapGestures[i]];
      
      padViews[i].hidden = NO;
      [padViews[i] removeFromSuperview];
      [scrollViews[i] addSubview : padViews[i]];
   }
   
   if (activeDemo) {
      scrollViews[activeView].hidden = NO;
      [self prepareHints];
   }
}

//_________________________________________________________________
- (IBAction) selectButtonPressed
{
   if (appMode == kTASelect)// || !activeDemo)
      return;

   appMode = kTASelect;

   //hide both scroll views, re-parent pad-views.
   for (unsigned i = 0; i < 2; ++i) { // < kTDNOfPads
      scrollViews[i].hidden = YES;
      //1. Check, if views must be resized (unscaled).
      [self resizePadView : i];

      [padViews[i] removeFromSuperview];
      
      padPanGestures[i] = [[UIPanGestureRecognizer alloc] initWithTarget:padViews[i] action:@selector(handlePanGesture:)];
      [padViews[i] addGestureRecognizer:padPanGestures[i]];
      
      padTapGestures[i] = [[UITapGestureRecognizer alloc] initWithTarget:padViews[i] action:@selector(handleTapGesture:)];
      [padViews[i] addGestureRecognizer:padTapGestures[i]];

      [padViews[i] setSelectionView:selectionViews[i]];
   
      [parentView addSubview:padViews[i]];
 
      if (activeDemo) //In case no demo was selected - nothing to show yet.
         padViews[i].hidden = i == activeView ? NO : YES;
   }

   if (activeDemo) {
      [padViews[activeView] setProcessPan:activeDemo->Supports3DRotation()];
      [padViews[activeView] setProcessTap:!activeDemo->Supports3DRotation()];
      [self prepareHints];
   }
}

//_________________________________________________________________
- (IBAction) editButtonPressed : (id) sender
{
   if (editorPopover && editorPopover.popoverVisible) {
      [editorPopover dismissPopoverAnimated : YES];
      return;
   } else {
      if (!padController) {
         padController = [[PadOptionsController alloc] initWithNibName : @"PadOptionsController" bundle : nil];
         padController.contentSizeForViewInPopover = CGSizeMake(250.f, 650.f);
      }

      if (!editorPopover) {
         editorPopover = [[UIPopoverController alloc] initWithContentViewController : padController];
         editorPopover.popoverContentSize = CGSizeMake(250.f, 650.f);
      }

      [editorPopover presentPopoverFromBarButtonItem : sender permittedArrowDirections:UIPopoverArrowDirectionAny animated : YES];
      [padController setView : padViews[activeView] andPad : pad];
   }
}

//_________________________________________________________________
- (IBAction) showHelp
{
   CATransition *transition = [CATransition animation];
   transition.duration = 0.25;
   transition.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
   transition.type = kCATransitionReveal;
   transition.subtype = kCATransitionFade;
   help.hidden = !help.hidden;
   [help.layer addAnimation:transition forKey:nil];
}

#pragma mark - UIScrollView's delegate.

//_________________________________________________________________
- (UIView *) viewForZoomingInScrollView : (UIScrollView *)scrollView
{
   if (scrollView == scrollViews[0])
      return padViews[0];

   return padViews[1];
}

//_________________________________________________________________
- (CGRect) centeredFrameForScrollView : (UIScrollView *)scroll andUIView : (UIView *)rView 
{
   CGSize boundsSize = scroll.bounds.size;
   CGRect frameToCenter = rView.frame;
   // center horizontally
   if (frameToCenter.size.width < boundsSize.width)
      frameToCenter.origin.x = (boundsSize.width - frameToCenter.size.width) / 2;
   else
      frameToCenter.origin.x = 0;
   // center vertically
   if (frameToCenter.size.height < boundsSize.height)
      frameToCenter.origin.y = (boundsSize.height - frameToCenter.size.height) / 2;
   else
      frameToCenter.origin.y = 0;

   return frameToCenter;
}

//_________________________________________________________________
- (void)scrollViewDidEndZooming:(UIScrollView *)scrollView withView:(UIView *)view atScale:(float)scale
{
   const CGPoint off = [scrollView contentOffset];
   CGRect oldRect = padViews[activeView].frame;
   oldRect.origin.x = 0.f;
   oldRect.origin.y = 0.f;
   
   if (abs(oldSizes.width - oldRect.size.width) < 0.01 && (abs(oldSizes.height - oldRect.size.height) < 0.01))
      return;

   oldSizes = oldRect.size;
   
   [padViews[activeView] removeFromSuperview];
   padViews[activeView] = [[PadView alloc] initWithFrame : oldRect forPad : pad];
   [scrollView addSubview:padViews[activeView]];
  
   [scrollView setZoomScale:1.f];
   scrollView.contentSize = oldRect.size;   
   scrollView.contentOffset = off;

   scrollView.minimumZoomScale = 640.f / oldRect.size.width;
   scrollView.maximumZoomScale = 1280.f / oldRect.size.width;
}

//_________________________________________________________________
- (void)scrollViewDidZoom:(UIScrollView *)scrollView 
{
   padViews[activeView].frame = [self centeredFrameForScrollView:scrollView andUIView:padViews[activeView]];
}

#pragma mark - Tab bar delegate.

//_________________________________________________________________
- (void) tabBar : (UITabBar *) tb didSelectItem:(UITabBarItem *)item
{
   if (item.tag == 1)
      [self zoomButtonPressed];
   else
      [self selectButtonPressed];
}

#pragma mark - Tap gesture handler.

//_________________________________________________________________
- (void) handleDoubleTapPad:(UITapGestureRecognizer *)tap
{
   if (appMode != kTAZoom || !activeDemo)
      return;

   if (oldSizes.width > 640.f)
      [self resizePadView : activeView];
   else {
      //Zoom to maximum.
      oldSizes = CGSizeMake(1280.f, 1280.f);
      CGRect newRect = CGRectMake(0.f, 0.f, 1280.f, 1280.f);
      
      [padViews[activeView] removeFromSuperview];

      padViews[activeView] = [[PadView alloc] initWithFrame : newRect forPad : pad];
      [scrollViews[activeView] addSubview:padViews[activeView]];
  
      [scrollViews[activeView] setZoomScale:1.f];
      scrollViews[activeView].contentSize = newRect.size;

      scrollViews[activeView].minimumZoomScale = 1.f;
      scrollViews[activeView].maximumZoomScale = 1.f;

      const CGPoint tapXY = [tap locationInView : tap.view];  
      scrollViews[activeView].contentOffset = CGPointMake(tapXY.x, tapXY.y);    
   }
}

#pragma mark - Hints.

//_________________________________________________________________
- (void) showPinchHint
{
   [hintView setHintIcon : @"pinch_gesture_icon.png" hintText : @"Use a pinch gesture to zoom/unzoom a pad. Tap to close this hint."];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showPanHint
{
   [hintView setHintIcon : @"pan_gesture_icon.png" hintText : @"When a pad zoomed in, use a pan gesture to scroll a pad. Tap to close this hint."];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showDoubleTapHint
{
   [hintView setHintIcon : @"double_tap_gesture_icon.png" hintText : @"Use the double tap gesture to zoom/unzoom a pad. Tap to close this hint."];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showRotationHint
{
   [hintView setHintIcon:@"rotate_icon.png" hintText : @"You can rotate 3D object, using pan gesture. Tap to close this hint."];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showSingleTapHint
{
   [hintView setHintIcon:@"single_tap_icon.png" hintText : @"Use a single tap gesture to select pad's contents. Tap to close this hint."];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

@end
