#import "ROOTObjectController.h"
#import "H1BinsInspector.h"
#import "RangeSlider.h"

#import "IOSFileScanner.h"
#import "TAxis.h"
#import "TH1.h"

@implementation H1BinsInspector {
   RangeSlider *axisRangeSlider;
   __weak ROOTObjectController *controller;
   TH1 *object;
}

//____________________________________________________________________________________________________
- (id) initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   if (self) {
      [self view];
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

   axisRangeSlider = [[RangeSlider alloc] initWithFrame : CGRectMake(0.f, 210.f, 250.f, 60.f)];
   [self.view addSubview : axisRangeSlider];
   
   [axisRangeSlider addTarget:self action:@selector(axisRangeChanged) forControlEvents : UIControlEventValueChanged];
}

//____________________________________________________________________________________________________
- (void) viewDidUnload
{
   [super viewDidUnload];
   // Release any retained subviews of the main view.
   // e.g. self.myOutlet = nil;
}

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation
{
   // Return YES for supported orientations
	return YES;
}

//____________________________________________________________________________________________________
-(void) axisRangeChanged
{
   minLabel.center = CGPointMake([axisRangeSlider getMinThumbX], minLabel.center.y);
   minLabel.text = [NSString stringWithFormat:@"%.3g", axisRangeSlider.selectedMinimumValue];
   maxLabel.center = CGPointMake([axisRangeSlider getMaxThumbX], maxLabel.center.y);
   maxLabel.text = [NSString stringWithFormat:@"%.3g", axisRangeSlider.selectedMaximumValue];
   
   //Update the histogram.
   object->GetXaxis()->SetRangeUser(axisRangeSlider.selectedMinimumValue, axisRangeSlider.selectedMaximumValue);
   [controller objectWasModifiedUpdateSelection : YES];
}

#pragma mark - ObjectInspectorComponent protocol.
//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = static_cast<TH1 *>(o);
   //
   const char *histTitle = object->GetTitle();
   if (!histTitle || !*histTitle)
      titleField.text = @"";
   else
      titleField.text = [NSString stringWithFormat : @"%s", histTitle];

   const TAxis *xAxis = object->GetXaxis();
   const unsigned nBins = xAxis->GetNbins();

   const double xMin = xAxis->GetBinLowEdge(1);
   minLabel.text = [NSString stringWithFormat : @"%.3g", xMin];
   const double xMax = xAxis->GetBinUpEdge(nBins);
   maxLabel.text = [NSString stringWithFormat : @"%.3g", xMax];
   
   [axisRangeSlider setSliderMin : xMin max : xMax selectedMin : xMin selectedMax : xMax];
   minLabel.center = CGPointMake([axisRangeSlider getMinThumbX], minLabel.center.y);
   maxLabel.center = CGPointMake([axisRangeSlider getMaxThumbX], maxLabel.center.y);
}

#pragma mark - GUI actions.

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
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
//   showMarkers.on ? controller.markerDrawOption = @"P" : controller.markerDrawOption = @"";
   [controller setMarker : showMarkers.on];
   [controller objectWasModifiedUpdateSelection : YES];
}


@end
