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
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

	return YES;
}

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   UIViewController<ObjectInspectorComponent> *rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) obj
{
   assert(obj != nullptr && "setObject:, parameter 'obj' is null");

   UIViewController<ObjectInspectorComponent> *rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setObject : obj];
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
