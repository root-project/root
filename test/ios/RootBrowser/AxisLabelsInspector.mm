#import "ROOTObjectController.h"
#import "AxisLabelsInspector.h"
#import "AxisColorInspector.h"
#import "AxisFontInspector.h"

#import "TObject.h"
#import "TAxis.h"

static const float sizeStep = 0.01f;
static const float minSize = 0.f;
static const float maxSize = 1.f;

static const float offsetStep = 0.001f;
static const float minOffset = -1.f;
static const float maxOffset = 1.f;

static const float totalHeight = 400.f;
static const float tabBarHeight = 49.f;
static const CGRect componentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation AxisLabelsInspector

@synthesize plusSize;
@synthesize minusSize;
@synthesize sizeLabel;
@synthesize plusOffset;
@synthesize minusOffset;
@synthesize offsetLabel;
@synthesize noExp;

//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return componentFrame;
}

//____________________________________________________________________________________________________
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
   
   [self view];
   
   if (self)
      [self view];

   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   self.plusSize = nil;
   self.minusSize = nil;
   self.sizeLabel = nil;
   self.plusOffset = nil;
   self.minusOffset = nil;
   self.offsetLabel = nil;
   self.noExp = nil;
   
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
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = dynamic_cast<TAxis *>(o);
   
   sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize()];
   offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset()];
   
   noExp.on = object->GetNoExponent();
}

//____________________________________________________________________________________________________
- (void) showLabelFontInspector
{
   AxisFontInspector *fontInspector = [[AxisFontInspector alloc] initWithNibName : @"AxisFontInspector" mode : ROOT_IOSObjectInspector::afimLabelFont];

   [fontInspector setROOTObjectController : controller];
   [fontInspector setROOTObject : object];
   
   [self.navigationController pushViewController : fontInspector animated : YES];
   [fontInspector release];
}

//____________________________________________________________________________________________________
- (IBAction) plusBtn : (UIButton *)sender
{
   if (sender == plusSize) {
      if (object->GetLabelSize() + sizeStep > maxSize)
         return;
      
      sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize() + sizeStep];
      object->SetLabelSize(object->GetLabelSize() + sizeStep);
   } else if (sender == plusOffset) {
      if (object->GetLabelOffset() + offsetStep > maxOffset)
         return;
      
      offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset() + offsetStep];
      object->SetLabelOffset(object->GetLabelOffset() + offsetStep);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusBtn : (UIButton *)sender
{
   if (sender == minusSize) {
      if (object->GetLabelSize() - sizeStep < minSize)
         return;
      
      sizeLabel.text = [NSString stringWithFormat : @"%.2f", object->GetLabelSize() - sizeStep];
      object->SetLabelSize(object->GetLabelSize() - sizeStep);
   } else if (sender == minusOffset) {
      if (object->GetLabelOffset() - offsetStep < minOffset)
         return;
      
      offsetLabel.text = [NSString stringWithFormat : @"%.3f", object->GetLabelOffset() - offsetStep];
      object->SetLabelOffset(object->GetLabelOffset() - offsetStep);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) noExpPressed
{
   object->SetNoExponent(noExp.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) back
{
   [self.navigationController popViewControllerAnimated : YES];
}

@end
