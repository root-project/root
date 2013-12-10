#import <cassert>

#import "ObjectViewController.h"
#import "H1BinsInspector.h"
#import "RangeSlider.h"

#import "TAxis.h"
#import "TH1.h"

@implementation H1BinsInspector {
   __weak IBOutlet UITextField *titleField;
   __weak IBOutlet UILabel *minLabel;
   __weak IBOutlet UILabel *maxLabel;
   __weak IBOutlet UISwitch *showMarkers;

   RangeSlider *axisRangeSlider;
   __weak ObjectViewController *controller;
   TH1 *object;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   if (self)//Force views load (not necessary??).
      [self view];

   return self;
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void) viewDidLoad
{
   [super viewDidLoad];

   axisRangeSlider = [[RangeSlider alloc] initWithFrame : CGRectMake(0.f, 210.f, 250.f, 60.f)];
   [self.view addSubview : axisRangeSlider];
   
   [axisRangeSlider addTarget:self action:@selector(axisRangeChanged) forControlEvents : UIControlEventValueChanged];
}

#pragma mark - interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)
	return YES;
}

#pragma mark - ObjectInspectorComponent protocol.


//____________________________________________________________________________________________________
-(void) axisRangeChanged
{
   assert(object != nullptr && "axisRangeChanged, object is null");

   minLabel.center = CGPointMake([axisRangeSlider getMinThumbX], minLabel.center.y);
   minLabel.text = [NSString stringWithFormat : @"%.3g", axisRangeSlider.selectedMinimumValue];
   maxLabel.center = CGPointMake([axisRangeSlider getMaxThumbX], maxLabel.center.y);
   maxLabel.text = [NSString stringWithFormat : @"%.3g", axisRangeSlider.selectedMaximumValue];
   
   //Update the histogram.
   object->GetXaxis()->SetRangeUser(axisRangeSlider.selectedMinimumValue, axisRangeSlider.selectedMaximumValue);
   [controller objectWasModifiedUpdateSelection : YES];
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

   object = static_cast<TH1 *>(o);
   //I do not use dynamic cast.
   const char *histTitle = object->GetTitle();
   if (!histTitle || !*histTitle)
      titleField.text = @"";
   else
      titleField.text = [NSString stringWithFormat : @"%s", histTitle];

   const TAxis *xAxis = object->GetXaxis();
   const unsigned nBins = xAxis->GetNbins();

   const double xMin = xAxis->GetBinLowEdge(1);
   const double xMinSelected = xAxis->GetBinLowEdge(xAxis->GetFirst());
   minLabel.text = [NSString stringWithFormat : @"%.3g", xMinSelected];
   const double xMax = xAxis->GetBinUpEdge(nBins);
   const double xMaxSelected = xAxis->GetBinUpEdge(xAxis->GetLast());
   maxLabel.text = [NSString stringWithFormat : @"%.3g", xMaxSelected];
   
   [axisRangeSlider setSliderMin : xMin max : xMax selectedMin : xMinSelected selectedMax : xMaxSelected];
   minLabel.center = CGPointMake([axisRangeSlider getMinThumbX], minLabel.center.y);
   maxLabel.center = CGPointMake([axisRangeSlider getMaxThumbX], maxLabel.center.y);
}

#pragma mark - GUI actions.

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
   assert(object != nullptr && "textFieldDidEndOnExit:, object is null");
   
   object->SetTitle([titleField.text cStringUsingEncoding : [NSString defaultCStringEncoding]]);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldEditingDidEnd : (id) sender
{
   [sender resignFirstResponder];
}

//____________________________________________________________________________________________________
- (IBAction) toggleMarkers
{
   [controller setMarker : showMarkers.on];
   [controller objectWasModifiedUpdateSelection : YES];
}


@end
