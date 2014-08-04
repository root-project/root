#import "InspectorWithNavigation.h"

@implementation InspectorWithNavigation

//____________________________________________________________________________________________________
- (id) initWithRootViewController : (UIViewController<ObjectInspectorComponent> *) rootController
{
   self = [super initWithRootViewController : (UIViewController *)rootController];
   if (self) {
      self.navigationBar.hidden = YES;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

/*
// Implement loadView to create a view hierarchy programmatically, without using a nib.
- (void)loadView
{
}
*/

/*
// Implement viewDidLoad to do additional setup after loading the view, typically from a nib.
- (void)viewDidLoad
{
    [super viewDidLoad];
}
*/

//____________________________________________________________________________________________________
- (void)viewDidUnload
{
   [super viewDidUnload];
   // Release any retained subviews of the main view.
   // e.g. self.myOutlet = nil;
}

//____________________________________________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
   // Return YES for supported orientations
   return YES;
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   UIViewController<ObjectInspectorComponent> *rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setROOTObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)obj
{
   UIViewController<ObjectInspectorComponent> *rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setROOTObject : obj];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   UIViewController<ObjectInspectorComponent> *rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   return [rootController getComponentName];
}

//____________________________________________________________________________________________________
- (void) resetInspector
{
   //Pop all controllers from a stack except a top level root controller.
   while ([self.viewControllers count] > 1)
      [self popViewControllerAnimated : NO];
}

@end
