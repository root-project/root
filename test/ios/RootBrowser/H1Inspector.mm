#import "H1ErrorsInspector.h"
#import "H1BinsInspector.h"
#import "H1Inspector.h"

//It's mm file, C++ constants have internal linkage.
const CGFloat totalHeight = 399.f;
const CGFloat tabBarHeight = 49.f;
const CGRect nestedComponentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@interface H1Inspector () {
   H1ErrorsInspector *errorInspector;
   H1BinsInspector *binsInspector;
   
   TObject *object;
   __weak ROOTObjectController *controller;
}

- (void) showBinsInspector;
- (void) showErrorInspector;

@end

@implementation H1Inspector 

//____________________________________________________________________________________________________
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
   
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

#pragma mark - ObjectInspectorComponent protocol.

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = o;
   [errorInspector setROOTObject : o];
   [binsInspector setROOTObject : o];
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
   [errorInspector setROOTObjectController : c];
   [binsInspector setROOTObjectController : c];
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
- (void) tabBar : (UITabBar *) tb didSelectItem : (UITabBarItem *)item
{
   if (item.tag == 1)
      [self showBinsInspector];
   else if (item.tag == 2)
      [self showErrorInspector];
}



@end
