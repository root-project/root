#import <cassert>

#import "InspectorWithNavigation.h"

@implementation InspectorWithNavigation

//____________________________________________________________________________________________________
- (instancetype) initWithRootViewController : (UIViewController<ObjectInspectorComponent> *) rootController
{
   self = [super initWithRootViewController : rootController];

   if (self)
      self.navigationBar.hidden = YES;
   
   return self;
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)
   return YES;
}

#pragma mark - Object inspector component's protocol.

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   UIViewController<ObjectInspectorComponent> * const rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) obj
{
   assert(obj != nullptr && "setObject:, parameter 'obj' is null");

   UIViewController<ObjectInspectorComponent> * const rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
   [rootController setObject : obj];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   UIViewController<ObjectInspectorComponent> * const rootController = (UIViewController<ObjectInspectorComponent> *)[self.viewControllers objectAtIndex : 0];
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
