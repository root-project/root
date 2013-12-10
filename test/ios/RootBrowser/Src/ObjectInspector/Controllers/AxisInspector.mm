#import <cassert>

#import "InspectorWithNavigation.h"
#import "AxisLabelsInspector.h"
#import "AxisTitleInspector.h"
#import "AxisTicksInspector.h"
#import "AxisInspector.h"

@implementation AxisInspector {
   __weak IBOutlet UITabBar *tabBar;

   AxisTicksInspector *ticksInspector;

   InspectorWithNavigation *titleInspector;
   InspectorWithNavigation *labelInspector;

   __weak ObjectViewController *controller;
   TObject *object;
}

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, 0.f, 250.f, 400.f);
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle:nibBundleOrNil];

   if (self) {
      [self view];//Force load. Do not want to wait.

      ticksInspector = [[AxisTicksInspector alloc] initWithNibName : @"AxisTicksInspector" bundle : nil];
      [self.view addSubview : ticksInspector.view];
   
      //
      AxisTitleInspector *titleInspectorCompositor = [[AxisTitleInspector alloc] initWithNibName : @"AxisTitleInspector" bundle : nil];
      //Title inspector is the first in the nested navigation stack.
      titleInspector = [[InspectorWithNavigation alloc] initWithRootViewController : titleInspectorCompositor];
      titleInspector.view.frame = [AxisTitleInspector inspectorFrame];
      [self.view addSubview : titleInspector.view];
      titleInspector.view.hidden = YES;
      //
      
      AxisLabelsInspector *labelInspectorCompositor = [[AxisLabelsInspector alloc] initWithNibName : @"AxisLabelsInspector" bundle : nil];
      //Label inspector is the first in the nested navigation stack.      
      labelInspector = [[InspectorWithNavigation alloc] initWithRootViewController : labelInspectorCompositor];
      labelInspector.view.frame = [AxisLabelsInspector inspectorFrame];
      [self.view addSubview : labelInspector.view];
      labelInspector.view.hidden = YES;
   
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

#pragma mark - ObjectInspectorComponent's protocol.
//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   controller = c;
   [ticksInspector setObjectController : c];
   [titleInspector setObjectController : c];
   [labelInspector setObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = o;
   [ticksInspector setObject : o];
   [titleInspector setObject : o];
   [labelInspector setObject : o];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Axis attributes";
}

//____________________________________________________________________________________________________
- (void) resetInspector
{
   tabBar.selectedItem = [[tabBar items] objectAtIndex : 0];
   [titleInspector resetInspector];
   [labelInspector resetInspector];
   
   [self showTicksInspector];
}

//____________________________________________________________________________________________________
- (void) showTicksInspector
{
   ticksInspector.view.hidden = NO;
   titleInspector.view.hidden = YES;
   labelInspector.view.hidden = YES;
}

//____________________________________________________________________________________________________
- (void) showAxisTitleInspector
{
   ticksInspector.view.hidden = YES;
   titleInspector.view.hidden = NO;
   labelInspector.view.hidden = YES;
}

//____________________________________________________________________________________________________
- (void) showAxisLabelsInspector
{
   ticksInspector.view.hidden = YES;
   titleInspector.view.hidden = YES;
   labelInspector.view.hidden = NO;
}

#pragma mark - Tabbar delegate.

//____________________________________________________________________________________________________
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *) item
{
#pragma unused(tb)

   if (item.tag == 1)
      [self showTicksInspector];
   else if (item.tag == 2)
      [self showAxisTitleInspector];
   else if (item.tag == 3)
      [self showAxisLabelsInspector];
}

@end
