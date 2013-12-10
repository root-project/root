#import <cassert>

#import "PadTicksGridInspector.h"
#import "PadLogScaleInspector.h"
#import "PadInspector.h"

const CGFloat totalHeight = 250.f;
const CGFloat tabBarHeight = 49.f;
const CGRect nestedComponentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation PadInspector {
   __weak IBOutlet UITabBar *tabBar;
   PadTicksGridInspector *gridInspector;
   PadLogScaleInspector *logScaleInspector;
   
   __weak ObjectViewController *controller;
   TObject *object;
}

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, 0.f, 250.f, 250.f);
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
    

    
   if (self) {
      //Force views load.
      [self view];
      //Load inspectors from nib files.
      gridInspector = [[PadTicksGridInspector alloc] initWithNibName : @"PadTicksGridInspector" bundle : nil];
      gridInspector.view.frame = nestedComponentFrame;
      logScaleInspector = [[PadLogScaleInspector alloc] initWithNibName : @"PadLogScaleInspector" bundle : nil];
      logScaleInspector.view.frame = nestedComponentFrame;
      
      [self.view addSubview : gridInspector.view];
      [self.view addSubview : logScaleInspector.view];
      
      gridInspector.view.hidden = NO;
      logScaleInspector.view.hidden = YES;
      
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

#pragma mark - ObjectInspectorComponent.

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   controller = c;
   [gridInspector setObjectController : c];
   [logScaleInspector setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = o;
   [gridInspector setObject : o];
   [logScaleInspector setObject : o];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Pad attributes";
}

//____________________________________________________________________________________________________
- (void) resetInspector
{
   tabBar.selectedItem = [[tabBar items] objectAtIndex : 0];
   [self showTicksAndGridInspector];
}

//____________________________________________________________________________________________________
- (void) showTicksAndGridInspector
{
   logScaleInspector.view.hidden = YES;
   gridInspector.view.hidden = NO;
}

//____________________________________________________________________________________________________
- (void) showLogScaleInspector
{
   logScaleInspector.view.hidden = NO;
   gridInspector.view.hidden = YES;
}

#pragma mark - Tabbar delegate.

//____________________________________________________________________________________________________
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *) item
{
#pragma unused(tb)

   if (item.tag == 1)
      [self showTicksAndGridInspector];
   else if (item.tag == 2)
      [self showLogScaleInspector];
}

@end
