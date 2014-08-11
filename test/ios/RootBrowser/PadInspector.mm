#import "PadTicksGridInspector.h"
#import "PadLogScaleInspector.h"
#import "PadInspector.h"

//It's mm file == C++, consts have internal linkage.
const CGFloat totalHeight = 250.f;
const CGFloat tabBarHeight = 49.f;
const CGRect nestedComponentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@interface PadInspector () {
   PadTicksGridInspector *gridInspector;
   PadLogScaleInspector *logScaleInspector;

   __weak ROOTObjectController *controller;
   TObject *object;
}

- (void) showTicksAndGridInspector;
- (void) showLogScaleInspector;

@end

@implementation PadInspector

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, 0.f, 250.f, 250.f);
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];

   [self view];

   if (self) {
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

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
   [gridInspector setROOTObjectController : c];
   [logScaleInspector setROOTObjectController : c];
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = o;
   [gridInspector setROOTObject : o];
   [logScaleInspector setROOTObject : o];
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
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *)item
{
   if (item.tag == 1)
      [self showTicksAndGridInspector];
   else if (item.tag == 2)
      [self showLogScaleInspector];
}

@end
