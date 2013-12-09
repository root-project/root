#import <cassert>
#import <cstring>

#import "ObjectViewController.h"
#import "AxisTicksInspector.h"

#import "TObject.h"
#import "TAxis.h"


//It's mm file == C++, consts have internal linkage.
const float tickLengthStep = 0.01f;
const float maxTickLength = 1.f;
const float minTickLength = -1.f;

const CGFloat tabBarHeight = 49.f;
const CGFloat totalHeight = 400.f;
const CGRect componentFrame = CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);

@implementation AxisTicksInspector {
   __weak IBOutlet UILabel *tickLengthLabel;
   __weak IBOutlet UIButton *plusLengthBtn;
   __weak IBOutlet UIButton *minusLengthBtn;

   __weak IBOutlet UIButton *plusPrim;
   __weak IBOutlet UIButton *minusPrim;
   __weak IBOutlet UILabel *primLabel;

   __weak IBOutlet UIButton *plusSec;
   __weak IBOutlet UIButton *minusSec;
   __weak IBOutlet UILabel *secLabel;

   __weak IBOutlet UIButton *plusTer;
   __weak IBOutlet UIButton *minusTer;
   __weak IBOutlet UILabel *terLabel;

   __weak IBOutlet UISegmentedControl *ticksNegPos;


   float tickLength;
   unsigned primaryTicks;
   unsigned secondaryTicks;
   unsigned tertiaryTicks;
   
   __weak ObjectViewController *controller;

   TAxis *object;
}

//____________________________________________________________________________________________________
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
   
   [self view];
   
   if (self) {
      // Custom initialization
      self.view.frame = componentFrame;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void) didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];
   // Do any additional setup after loading the view from its nib.
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

   return YES;
}

#pragma mark - Inspector's protocol.

//____________________________________________________________________________________________________
- (void) setTicksWidgets
{
//   object->SetNdivisions(int(primaryTicks.value), int(secondaryTicks.value), int(tertiaryTicks.value), 1);
//   [controller objectWasModifiedByEditor];

   const int nDivisions = object->GetNdivisions();
   //Hardcoded constants from TAttAxis, never defined as named values in ROOT.
   //"Formulas" from TAxisEditor.
   primaryTicks = nDivisions % 100;
   secondaryTicks = (nDivisions / 100) % 100;
   tertiaryTicks = (nDivisions / 10000) % 100;
   
   primLabel.text = [NSString stringWithFormat : @"%u", primaryTicks];
   secLabel.text = [NSString stringWithFormat : @"%u", secondaryTicks];
   terLabel.text = [NSString stringWithFormat : @"%u", tertiaryTicks];
   
   tickLength = object->GetTickLength();
   tickLengthLabel.text = [NSString stringWithFormat : @"%.2f", object->GetTickLength()];
}

//____________________________________________________________________________________________________
- (void) setupInspector
{
   const char *option = object->GetTicks();
   
   if (!std::strcmp("+-", option))
      [ticksNegPos setSelectedSegmentIndex : 1];
   else 
      [ticksNegPos setSelectedSegmentIndex : 0];

   [self setTicksWidgets];
}

//____________________________________________________________________________________________________
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   controller = c;
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   //The result of a cast is not checked here, it's done on top level.
   object = dynamic_cast<TAxis *>(o);

   [self setupInspector];
}

//____________________________________________________________________________________________________
- (IBAction) ticksNegPosPressed
{
   if (ticksNegPos.selectedSegmentIndex == 0)
      object->SetTicks("");
   else
      object->SetTicks("+-");

   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (void) setTicks
{
   object->SetNdivisions(primaryTicks, secondaryTicks, tertiaryTicks);
}

//____________________________________________________________________________________________________
- (IBAction) plusTick : (UIButton *)sender
{
   if (sender == plusLengthBtn) {
      if (tickLength + tickLengthStep < maxTickLength) {
         tickLength += tickLengthStep;
         tickLengthLabel.text = [NSString stringWithFormat:@"%.2f", tickLength];
         object->SetTickLength(tickLength);
         [controller objectWasModifiedUpdateSelection : NO];
      }
      return;
   }
   
   UILabel *labelToModify = 0;
   unsigned n = 0;

   if (sender == plusPrim) {
      labelToModify = primLabel;
      if (primaryTicks < 99)
         n = ++primaryTicks;
      else
         return;
   } else if (sender == plusSec) {
      labelToModify = secLabel;
      if (secondaryTicks < 99)
         n = ++secondaryTicks;
      else
         return;
   } else {
      labelToModify = terLabel;
      if (tertiaryTicks < 99)
         n = ++tertiaryTicks;
      else
         return;
   }
   
   labelToModify.text = [NSString stringWithFormat : @"%u", n];
   [self setTicks];
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusTick : (UIButton *) sender
{
   if (sender == minusLengthBtn) {
      if (tickLength - tickLengthStep > minTickLength) {
         tickLength -= tickLengthStep;
         tickLengthLabel.text = [NSString stringWithFormat:@"%.2g", tickLength];
         object->SetTickLength(tickLength);
         [controller objectWasModifiedUpdateSelection : NO];
      }
      return;
   }

   UILabel *labelToModify = 0;
   unsigned n = 0;

   if (sender == minusPrim) {
      labelToModify = primLabel;
      if (primaryTicks > 0)
         n = --primaryTicks;
      else
         return;
   } else if (sender == minusSec) {
      labelToModify = secLabel;
      if (secondaryTicks > 0)
         n = --secondaryTicks;
      else
         return;
   } else {
      labelToModify = terLabel;
      if (tertiaryTicks > 0)
         n = --tertiaryTicks;
      else
         return;
   }
   
   labelToModify.text = [NSString stringWithFormat : @"%u", n];
   [self setTicks];
   [controller objectWasModifiedUpdateSelection : NO];
}


@end
