#import "InspectorWithNavigation.h"
#import "AxisLabelsInspector.h"
#import "AxisTitleInspector.h"
#import "AxisTicksInspector.h"
#import "AxisInspector.h"

@interface AxisInspector () {
   AxisTicksInspector *ticksInspector;

   InspectorWithNavigation *titleInspector;
   InspectorWithNavigation *labelInspector;

   __weak ROOTObjectController *controller;
   TObject *object;
}

- (void) showTicksInspector;
- (void) showAxisTitleInspector;
- (void) showAxisLabelsInspector;

@end


@implementation AxisInspector

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, 0.f, 250.f, 400.f);
}

//____________________________________________________________________________________________________
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle:nibBundleOrNil];

   if (self) {
      [self view];

      ticksInspector = [[AxisTicksInspector alloc] initWithNibName : @"AxisTicksInspector" bundle : nil];
      [self.view addSubview : ticksInspector.view];
   
      //
      AxisTitleInspector *titleInspectorCompositor = [[AxisTitleInspector alloc] initWithNibName : @"AxisTitleInspector" bundle : nil];
      titleInspector = [[InspectorWithNavigation alloc] initWithRootViewController : titleInspectorCompositor];
      titleInspector.view.frame = [AxisTitleInspector inspectorFrame];
      [self.view addSubview : titleInspector.view];
      titleInspector.view.hidden = YES;
      //
      
      AxisLabelsInspector *labelInspectorCompositor = [[AxisLabelsInspector alloc] initWithNibName : @"AxisLabelsInspector" bundle : nil];
      labelInspector = [[InspectorWithNavigation alloc] initWithRootViewController : labelInspectorCompositor];
      labelInspector.view.frame = [AxisLabelsInspector inspectorFrame];
      [self.view addSubview : labelInspector.view];
      labelInspector.view.hidden = YES;
   
      tabBar.selectedItem = [[tabBar items] objectAtIndex : 0];
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

//____________________________________________________________________________________________________
- (void)viewDidLoad
{
   [super viewDidLoad];
   // Do any additional setup after loading the view from its nib.
}

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

#pragma mark - ObjectInspectorComponent's protocol.
//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
   [ticksInspector setROOTObjectController : c];
   [titleInspector setROOTObjectController : c];
   [labelInspector setROOTObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = o;
   [ticksInspector setROOTObject : o];
   [titleInspector setROOTObject : o];
   [labelInspector setROOTObject : o];
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
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *)item
{
   if (item.tag == 1)
      [self showTicksInspector];
   else if (item.tag == 2)
      [self showAxisTitleInspector];
   else if (item.tag == 3)
      [self showAxisLabelsInspector];
}

@end
