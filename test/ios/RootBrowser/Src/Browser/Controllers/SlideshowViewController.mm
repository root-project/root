#import <cassert>

#import <QuartzCore/QuartzCore.h>

#import "SlideshowViewController.h"
#import "PadSlideView.h"

//C++ imports.
#import "IOSPad.h"

#import "FileUtils.h"

@implementation SlideshowViewController {
   __weak IBOutlet UIView *parentView;
   __weak IBOutlet UIView *padParentView;

   PadSlideView *padViews[2];//The current and the next in a slide show.

   unsigned visiblePad;
   unsigned nCurrentObject;
   
   ROOT::iOS::Browser::FileContainer *fileContainer;
   
   NSTimer *timer;

   BOOL viewDidAppear;
}

#pragma mark - Geometry/views/subviews.

//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
#pragma unused(orientation)

   const CGRect mainFrame = self.view.frame;
   CGRect padFrame = [PadSlideView slideFrame];
   padFrame.origin = CGPointMake(mainFrame.size.width / 2 - padFrame.size.width / 2, mainFrame.size.height / 2 - padFrame.size.height / 2);
   
   padParentView.frame = padFrame;

   if (padViews[0]) {
      padFrame.origin = CGPointZero;
      padViews[0].frame = padFrame;
      padViews[1].frame = padFrame;
   }
}

//____________________________________________________________________________________________________
- (void) initPadViews
{
   const CGRect padFrame = [PadSlideView slideFrame];

   unsigned nObjects = fileContainer->GetNumberOfObjects();
   if (nObjects > 2)
      nObjects = 2;

   for (unsigned i = 0; i < nObjects; ++i) {
      padViews[i] = [[PadSlideView alloc] initWithFrame : padFrame];
      [padParentView addSubview : padViews[i]];
      padViews[i].hidden = YES;
   }
}

#pragma mark - Fugly two-phase initialization.

//____________________________________________________________________________________________________
- (instancetype) initWithCoder : (NSCoder *) aDecoder
{
   if (self = [super initWithCoder : aDecoder]) {
      fileContainer = nullptr;
      viewDidAppear = NO;
   }
   
   return self;
}

//____________________________________________________________________________________________________
- (void) setFileContainer : (ROOT::iOS::Browser::FileContainer *) container
{
   assert(container != nullptr && "setFileContainer:, parameter 'container' is null");
   
   fileContainer = container;
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) dealloc
{
   if (timer)
      [timer invalidate];
}

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   assert(fileContainer != nullptr && "viewDidLoad, fileContainer is null");
   
   [super viewDidLoad];
   [self initPadViews];
}

//____________________________________________________________________________________________________
- (void) viewDidAppear : (BOOL) animated
{
   [super viewDidAppear : animated];

   assert(fileContainer != nullptr && "viewDidAppera:, fileContainer is null");

   if (!viewDidAppear) {
      if (fileContainer->GetNumberOfObjects()) {
         nCurrentObject = 0;
         visiblePad = 0;

         [padViews[0] setPad : fileContainer->GetPadAttached(0)];
         [padViews[0] setNeedsDisplay];

         if (fileContainer->GetNumberOfObjects() > 1) {
            [padViews[1] setPad : fileContainer->GetPadAttached(1)];
            [padViews[1] setNeedsDisplay];
         }

         padViews[0].hidden = NO;
      }

      if (fileContainer->GetNumberOfObjects() > 1)
         timer = [NSTimer scheduledTimerWithTimeInterval : 2.f target : self selector : @selector(changeViews) userInfo : nil repeats : YES];
      
      viewDidAppear = YES;
   }
}

//____________________________________________________________________________________________________
- (void) viewDidLayoutSubviews
{
   [self correctFramesForOrientation : self.interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void) viewDidDisappear : (BOOL)animated
{
   if (timer) {
      [timer invalidate];
      timer = 0;
   }
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)
   return YES;
}

//____________________________________________________________________________________________________
- (void) willAnimateRotationToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation duration : (NSTimeInterval) duration
{
#pragma unused(duration)
   [self correctFramesForOrientation : interfaceOrientation];
}

#pragma mark - Animation.

//____________________________________________________________________________________________________
- (void) changeViews
{
   const UIViewAnimationTransition animations[] = {UIViewAnimationTransitionFlipFromLeft, UIViewAnimationTransitionFlipFromRight, 
                                                   UIViewAnimationTransitionCurlUp, UIViewAnimationTransitionCurlDown};
   const UIViewAnimationTransition currentAnimation = animations[rand() % 4];

   const unsigned viewToHide = visiblePad;
   const unsigned viewToShow = !visiblePad;

   [UIView beginAnimations : @"hide view" context : nil];
   [UIView setAnimationDuration : 0.5];
   [UIView setAnimationCurve : UIViewAnimationCurveEaseInOut];
   [UIView setAnimationTransition : currentAnimation forView : padParentView cache : YES];

   padViews[viewToHide].hidden = YES;
   padViews[viewToShow].hidden = NO;
   
   [UIView commitAnimations];

   nCurrentObject + 1 == fileContainer->GetNumberOfObjects() ? nCurrentObject = 0 : ++nCurrentObject;
   visiblePad = viewToShow;
   const unsigned next = nCurrentObject + 1 == fileContainer->GetNumberOfObjects() ? 0 : nCurrentObject + 1;
   [padViews[viewToHide] setPad : fileContainer->GetPadAttached(next)];
   [padViews[viewToHide] setNeedsDisplay];
}

@end
