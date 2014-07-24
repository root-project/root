#import <cassert>

#import "H1ErrorsInspector.h"
#import "H1BinsInspector.h"
#import "H1Inspector.h"

//It's mm file, C++ constants have internal linkage.
const CGFloat totalHeight = 399.f;
const CGFloat tabBarHeight = 49.f;
const CGRect nestedComponentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation H1Inspector {
   __weak IBOutlet UITabBar *tabBar;
   
   H1ErrorsInspector *errorInspector;
   H1BinsInspector *binsInspector;
   
   TObject *object;
   __weak ObjectViewController *controller;

}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   
   if (self) {
      [self view];
      
      errorInspector = [[H1ErrorsInspector alloc] initWithNibName : @"H1ErrorsInspector" bundle : nil];
      errorInspector.view.frame = nestedComponentFrame;
      [self.view addSubview : errorInspector.view];
      errorInspector.view.hidden = YES;
      
      binsInspector = [[H1BinsInspector alloc] initWithNibName : @"H1BinsInspector" bundle : nil];
      binsInspector.view.frame = nestedComponentFrame;
      [self.view addSubview : binsInspector.view];
      binsInspector.view.hidden = NO;
      
      tabBar.selectedItem = [[tabBar items] objectAtIndex : 0];
   }
   
   return self;
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

   return YES;
}

#pragma mark - ObjectInspectorComponent protocol.

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   controller = c;
   [errorInspector setObjectController : c];
   [binsInspector setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = o;
   [errorInspector setObject : o];
   [binsInspector setObject : o];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Hist attributes";
}

//____________________________________________________________________________________________________
- (void) resetInspector
{
   tabBar.selectedItem = [[tabBar items] objectAtIndex : 0];
   [self showBinsInspector];
}

#pragma mark - Sub-components.

//____________________________________________________________________________________________________
- (void) showBinsInspector
{
   binsInspector.view.hidden = NO;
   errorInspector.view.hidden = YES;
}


//____________________________________________________________________________________________________
- (void) showErrorInspector
{
   binsInspector.view.hidden = YES;
   errorInspector.view.hidden = NO;
}

#pragma mark - UITabBar's delegate.

//____________________________________________________________________________________________________
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *) item
{
#pragma unused(tb)

   if (item.tag == 1)
      [self showBinsInspector];
   else if (item.tag == 2)
      [self showErrorInspector];
}



@end
