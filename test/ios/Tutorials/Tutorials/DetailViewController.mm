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

@interface DetailViewController ()
@property (nonatomic, retain) UIPopoverController *popoverController;
- (void)configureView;
@end

@implementation DetailViewController


@synthesize toolbar=_toolbar;
@synthesize detailItem=_detailItem;
@synthesize detailDescriptionLabel=_detailDescriptionLabel;
@synthesize popoverController=_myPopoverController;
@synthesize help;
@synthesize padController = padController_;
@synthesize editorPopover = editorPopover_;

#pragma mark - Managing the detail item

//_________________________________________________________________
- (void)setDetailItem:(id)newDetailItem
{
   //When setting the detail item, update the view and dismiss the popover controller if it's showing.
   
   if (_detailItem != newDetailItem) {
      [_detailItem release];
      _detailItem = [newDetailItem retain];
      
      
      // Update the view.
      [self configureView];
   }

   if (self.popoverController != nil)
      [self.popoverController dismissPopoverAnimated:YES];
}

//_________________________________________________________________
- (void)configureView
{
   // Update the user interface for the detail item.
   //self.detailDescriptionLabel.text = [self.detailItem description];
}

//_________________________________________________________________
- (void)initCPPObjects
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
   
   [parentView release];

   ///
   padRect.origin.x = 0.f, padRect.origin.y = 0.f;
   for (unsigned i = 0; i < 2; ++i) {// < kTDNOfPads
      scrollViews[i] = [[UIScrollView alloc] initWithFrame:padRect];
      scrollViews[i].backgroundColor = [UIColor darkGrayColor];
      scrollViews[i].delegate = self;
      padViews[i] = [[PadView alloc] initWithFrame : padRect forPad : pad];
      scrollViews[i].contentSize = padViews[i].frame.size;
      [scrollViews[i] addSubview:padViews[i]];
      [padViews[i] release];
      //
      scrollViews[i].minimumZoomScale = 1.f;
      scrollViews[i].maximumZoomScale = 1280.f / 640.f;
      [scrollViews[i] setZoomScale:1.f];
      [parentView addSubview:scrollViews[i]];
      [scrollViews[i] release];
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
   /*   selectionViews[i].layer.shadowColor = [UIColor blackColor].CGColor;
      selectionViews[i].layer.shadowOpacity = 0.9f;
      selectionViews[i].layer.shadowOffset = CGSizeMake(10.f, 10.f);*/
      
      [selectionViews[i] release];
   }
}

//_________________________________________________________________
- (void) initHints
{
   //Hints:
   pinchHintText = @"Use a pinch gesture to zoom/unzoom a pad. Tap to close this hint.";
   panHintText = @"When a pad zoomed in, use a pan gesture to scroll a pad. Tap to close this hint.";
   doubleTapHintText = @"Use the double tap gesture to zoom/unzoom a pad. Tap to close this hint.";
   singleTapHintText = @"Use a single tap gesture to select pad's contents. Tap to close this hint.";
   rotateHintText = @"You can rotate 3D object, using pan gesture. Tap to close this hint.";
//   emptyHintText = @"No gesture is available in the current mode for the current demo. Tap to close this hint.";

   //Pictogramms.
   CGRect pictRect = CGRectMake(670.f, 450.f, 50.f, 50.f);
   pinchPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"pinch_gesture_icon_small.png"];
   pinchPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:pinchPic];
   UITapGestureRecognizer *pinchTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showPinchHint)];
   [pinchPic addGestureRecognizer:pinchTap];
   [pinchTap release];
   pinchPic.hidden = YES;
   [pinchPic release];

   pictRect.origin.y = 520;
   panPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"pan_gesture_icon_small.png"];
   panPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:panPic];
   UITapGestureRecognizer *panTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showPanHint)];
   [panPic addGestureRecognizer:panTap];
   [panTap release];
   panPic.hidden = YES;
   [panPic release];
   
   pictRect.origin.y = 590;
   doubleTapPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"double_tap_gesture_icon_small.png"];
   doubleTapPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:doubleTapPic];
   UITapGestureRecognizer *dtapTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showDoubleTapHint)];
   [doubleTapPic addGestureRecognizer:dtapTap];
   [dtapTap release];
   doubleTapPic.hidden = YES;
   [doubleTapPic release];

   rotatePic = [[PictView alloc] initWithFrame:pictRect andIcon:@"rotate_icon_small.png"];
   rotatePic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:rotatePic];
   UITapGestureRecognizer *rotTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showRotationHint)];
   [rotatePic addGestureRecognizer:rotTap];
   [rotTap release];
   rotatePic.hidden = YES;
   [rotatePic release];
   
   singleTapPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"single_tap_icon_small.png"];
   singleTapPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:singleTapPic];
   UITapGestureRecognizer *singleTapTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showSingleTapHint)];
   [singleTapPic addGestureRecognizer:singleTapTap];
   [singleTapTap release];
   singleTapPic.hidden = YES;
   [singleTapPic release];


/*   emptyPic = [[PictView alloc] initWithFrame:pictRect andIcon:@"no_gesture_icon_small.png"];
   emptyPic.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:emptyPic];
   UITapGestureRecognizer *emTap = [[UITapGestureRecognizer alloc]initWithTarget:self action:@selector(showEmptyHint)];
   [emptyPic addGestureRecognizer:emTap];
   [emTap release];
   emptyPic.hidden = YES;
   [emptyPic release];*/
   
   const CGPoint center = self.view.center;
   CGRect rect = CGRectMake(center.x - 300.f, center.y - 290.f, 600.f, 600.f);
   hintView = [[HintView alloc] initWithFrame:rect];
   hintView.autoresizingMask = UIViewAutoresizingFlexibleBottomMargin | UIViewAutoresizingFlexibleTopMargin | UIViewAutoresizingFlexibleLeftMargin | UIViewAutoresizingFlexibleRightMargin;
   [self.view addSubview:hintView];
   UITapGestureRecognizer *hintTap = [[UITapGestureRecognizer alloc] initWithTarget:hintView action:@selector(handleTap:)];
   [hintView addGestureRecognizer:hintTap];
   [hintTap release];
   hintView.hidden = YES;
   [hintView release];
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
   [items release];
   self.popoverController = pc;
}

//_________________________________________________________________
- (void)splitViewController:(UISplitViewController *)svc willShowViewController:(UIViewController *)aViewController invalidatingBarButtonItem:(UIBarButtonItem *)barButtonItem
{
   // Called when the view is shown again in the split view, invalidating the button and popover controller.
   NSMutableArray *items = [[self.toolbar items] mutableCopy];
   [items removeObjectAtIndex:0];
   [self.toolbar setItems:items animated:YES];
   [items release];
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
   [tapGesture release];
      
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
   tb.selectedItem = [tb.items objectAtIndex:0];

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
   self.padController = nil;
   self.editorPopover = nil;
   self.help = nil;
   [_myPopoverController release];
   [_toolbar release];
   [_detailItem release];
   [_detailDescriptionLabel release];
   [super dealloc];
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
      [parentView.layer addAnimation:transition forKey:nil];
   }
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
   [padViews[view] release];
  
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
      [padViews[i] removeGestureRecognizer:padPanGestures[i]];
      [padViews[i] removeGestureRecognizer:padTapGestures[i]];
      
      [padViews[i] retain];
      padViews[i].hidden = NO;
      [padViews[i] removeFromSuperview];
      [scrollViews[i] addSubview:padViews[i]];
      [padViews[i] release];
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

      [padViews[i] retain];
      [padViews[i] removeFromSuperview];
      
      padPanGestures[i] = [[UIPanGestureRecognizer alloc] initWithTarget:padViews[i] action:@selector(handlePanGesture:)];
      [padViews[i] addGestureRecognizer:padPanGestures[i]];
      [padPanGestures[i] release];
      
      padTapGestures[i] = [[UITapGestureRecognizer alloc] initWithTarget:padViews[i] action:@selector(handleTapGesture:)];
      [padViews[i] addGestureRecognizer:padTapGestures[i]];
      [padTapGestures[i] release];

      [padViews[i] setSelectionView:selectionViews[i]];
   
      [parentView addSubview:padViews[i]];
      [padViews[i] release];
 
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
   if (editorPopover_ && editorPopover_.popoverVisible) {
      [editorPopover_ dismissPopoverAnimated : YES];
      return;
   } else {
      if (!padController_) {
         PadOptionsController *padInspector = [[PadOptionsController alloc] initWithNibName:@"PadOptionsController" bundle : nil];
         self.padController = padInspector;
         self.padController.contentSizeForViewInPopover = CGSizeMake(250.f, 650.f);
         [padInspector release];
      }

      if (!editorPopover_) {
         UIPopoverController *editorPopover = [[UIPopoverController alloc] initWithContentViewController : padController_];
         self.editorPopover = editorPopover;
         self.editorPopover.popoverContentSize = CGSizeMake(250.f, 650.f);
         [editorPopover release];
      }

      [self.editorPopover presentPopoverFromBarButtonItem : sender permittedArrowDirections:UIPopoverArrowDirectionAny animated : YES];
      [self.padController setView : padViews[activeView] andPad : pad];
   }
}

//DELEGATES:


//_________________________________________________________________
- (UIView *)viewForZoomingInScrollView:(UIScrollView *)scrollView
{
   if (scrollView == scrollViews[0])
      return padViews[0];
   return padViews[1];
}

//_________________________________________________________________
- (CGRect)centeredFrameForScrollView:(UIScrollView *)scroll andUIView:(UIView *)rView 
{
   CGSize boundsSize = scroll.bounds.size;
   CGRect frameToCenter = rView.frame;
   // center horizontally
   if (frameToCenter.size.width < boundsSize.width) {
      frameToCenter.origin.x = (boundsSize.width - frameToCenter.size.width) / 2;
   }
   else {
      frameToCenter.origin.x = 0;
   }
   // center vertically
   if (frameToCenter.size.height < boundsSize.height) {
      frameToCenter.origin.y = (boundsSize.height - frameToCenter.size.height) / 2;
   }
   else {
      frameToCenter.origin.y = 0;
   }
   
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
   [padViews[activeView] release];
  
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

//_________________________________________________________________
- (void) showPinchHint
{
   [hintView setHintIcon:@"pinch_gesture_icon.png" hintText:pinchHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showPanHint
{
   [hintView setHintIcon:@"pan_gesture_icon.png" hintText:panHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showDoubleTapHint
{
   [hintView setHintIcon:@"double_tap_gesture_icon.png" hintText:doubleTapHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showRotationHint
{
   [hintView setHintIcon:@"rotate_icon.png" hintText:rotateHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) showEmptyHint
{
/*   [hintView setHintIcon:@"no_gesture_icon.png" hintText:emptyHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;*/
}

//_________________________________________________________________
- (void) showSingleTapHint
{
   [hintView setHintIcon:@"single_tap_icon.png" hintText:singleTapHintText];
   [hintView setNeedsDisplay];
   hintView.hidden = NO;
}

//_________________________________________________________________
- (void) handleDoubleTapPad:(UITapGestureRecognizer *)tap
{
   if (appMode != kTAZoom || !activeDemo)
      return;

   if (oldSizes.width > 640.f)
      [self resizePadView: activeView];
   else {
      //Zoom to maximum.
      oldSizes = CGSizeMake(1280.f, 1280.f);
      CGRect newRect = CGRectMake(0.f, 0.f, 1280.f, 1280.f);
      
      [padViews[activeView] removeFromSuperview];

      padViews[activeView] = [[PadView alloc] initWithFrame : newRect forPad : pad];
      [scrollViews[activeView] addSubview:padViews[activeView]];
      [padViews[activeView] release];
  
      [scrollViews[activeView] setZoomScale:1.f];
      scrollViews[activeView].contentSize = newRect.size;

      scrollViews[activeView].minimumZoomScale = 1.f;
      scrollViews[activeView].maximumZoomScale = 1.f;

      const CGPoint tapXY = [tap locationInView : tap.view];  
      scrollViews[activeView].contentOffset = CGPointMake(tapXY.x, tapXY.y);    
   }
}

//_________________________________________________________________
- (void) showHelp
{
// First create a CATransition object to describe the transition
   CATransition *transition = [CATransition animation];
   // Animate over 3/4 of a second
   transition.duration = 0.25;
   // using the ease in/out timing function
   transition.timingFunction = [CAMediaTimingFunction functionWithName:kCAMediaTimingFunctionEaseInEaseOut];
   // Now to set the type of transition.
   transition.type = kCATransitionReveal;
   transition.subtype = kCATransitionFade;
   help.hidden = !help.hidden;
   // Finally, to avoid overlapping transitions we assign ourselves as the delegate for the animation and wait for the
   // -animationDidStop:finished: message. When it comes in, we will flag that we are no longer transitioning.
   //transitioning = YES;
   //transition.delegate = self;
   // Next add it to the containerView's layer. This will perform the transition based on how we change its contents.
   [help.layer addAnimation:transition forKey:nil];

}

//_________________________________________________________________
- (void) tabBar : (UITabBar *) tb didSelectItem:(UITabBarItem *)item
{
   if (item.tag == 1)
      [self zoomButtonPressed];
   else
      [self selectButtonPressed];
}

@end
