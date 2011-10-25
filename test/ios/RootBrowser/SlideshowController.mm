#import <stdlib.h>

#import <QuartzCore/QuartzCore.h>

#import "SlideshowController.h"
#import "SlideView.h"

//C++ (ROOT) imports.
#import "IOSFileContainer.h"
#import "IOSPad.h"

@implementation SlideshowController


@synthesize parentView;
@synthesize padParentView;

//____________________________________________________________________________________________________
- (void) correctFramesForOrientation : (UIInterfaceOrientation) orientation
{
   CGRect mainFrame;
   UIInterfaceOrientationIsPortrait(orientation) ? mainFrame = CGRectMake(0.f, 44.f, 768.f, 960.f)
                                                 : (mainFrame = CGRectMake(0.f, 44.f, 1024.f, 704.f));

   
   parentView.frame = mainFrame;
   
   CGRect padFrame = [SlideView slideFrame];
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
   const CGRect padFrame = [SlideView slideFrame];

   unsigned nObjects = fileContainer->GetNumberOfObjects();
   if (nObjects > 2)
      nObjects = 2;

   for (unsigned i = 0; i < nObjects; ++i) {
      padViews[i] = [[SlideView alloc] initWithFrame : padFrame];
      [padParentView addSubview : padViews[i]];
      padViews[i].hidden = YES;
      [padViews[i] release];
   }
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil fileContainer : (ROOT::iOS::FileContainer *)container
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self) {
      [self view];

      fileContainer = container;
      
      if (fileContainer->GetNumberOfObjects()) {
         [self initPadViews];

         nCurrentObject = 0;
         visiblePad = 0;

         [padViews[0] setPad : fileContainer->GetPadAttached(0)];
         [padViews[0] setNeedsDisplay];

         if (fileContainer->GetNumberOfObjects() > 1) {
            [padViews[1] setPad:fileContainer->GetPadAttached(1)];
            [padParentView addSubview : padViews[1]];
         }

         //Ready for show now.
      }
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   if (timer)
      [timer invalidate];

   self.parentView = nil;
   self.padParentView = nil;

   [super dealloc];
}

//____________________________________________________________________________________________________
- (void)didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void)viewDidLoad
{
   [super viewDidLoad];
   
   [self correctFramesForOrientation : self.interfaceOrientation];
}

//____________________________________________________________________________________________________
- (void)viewDidUnload
{
   [super viewDidUnload];
   // Release any retained subviews of the main view.
   // e.g. self.myOutlet = nil;
}

//____________________________________________________________________________________________________
- (void) viewWillAppear : (BOOL)animated
{
   [self correctFramesForOrientation : self.interfaceOrientation];
   padViews[0].hidden = NO;
}

//____________________________________________________________________________________________________
- (void) viewDidAppear : (BOOL)animated
{
   if (fileContainer->GetNumberOfObjects() > 1)
      timer = [NSTimer scheduledTimerWithTimeInterval : 2.f target : self selector : @selector(changeViews) userInfo : nil repeats : YES];
}


//____________________________________________________________________________________________________
- (void) viewDidDisappear:(BOOL)animated
{
   if (timer) {
      [timer invalidate];
      timer = 0;
   }
}

//____________________________________________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
   // Return YES for supported orientations
	
   return YES;
}

//____________________________________________________________________________________________________
- (void)willAnimateRotationToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation duration : (NSTimeInterval)duration {
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
