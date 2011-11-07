#import "ROOTObjectController.h"
#import "AxisLabelsInspector.h"
#import "AxisFontInspector.h"

#import "TObject.h"
#import "TAxis.h"

//It's mm file == C++, consts have internal linkage.
const float sizeStep = 0.01f;
const float minSize = 0.f;
const float maxSize = 1.f;

const float offsetStep = 0.001f;
const float minOffset = -1.f;
const float maxOffset = 1.f;

const float totalHeight = 400.f;
const float tabBarHeight = 49.f;
const CGRect componentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation AxisLabelsInspector {
   __weak ROOTObjectController *controller;
   TAxis *object;
}

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

@end
