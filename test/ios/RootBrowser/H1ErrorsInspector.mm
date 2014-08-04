#import <Availability.h>

#import "ROOTObjectController.h"
#import "H1ErrorsInspector.h"

//C++ imports.
#import "TH1.h"

#import "FileUtils.h"

namespace {

const CGFloat defaultCellW = 180.f;
const CGFloat defaultCellH = 44.f;



enum H1ErrorType {
   kNoError,
   kSimple,
   kEdges,
   kRectangles,
   kFill,
   kContour,
   kTotalNumOfTypes
};

NSString *errorTypesStrings[] = {@"No error", @"Simple", @"Edges", @"Rectangles", @"Fill", @"Contour"};

namespace RIB = ROOT::iOS::Browser;
RIB::EHistogramErrorOption histErrorTypes[] = {RIB::hetNoError, RIB::hetE, RIB::hetE1, RIB::hetE2, RIB::hetE3, RIB::hetE4};

}

@implementation H1ErrorsInspector {
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

#pragma mark - Pickerview delegate/data source.

//____________________________________________________________________________________________________
- (CGFloat)pickerView : (UIPickerView *)pickerView widthForComponent : (NSInteger)component
{
   return defaultCellW;
}

//____________________________________________________________________________________________________
- (CGFloat)pickerView : (UIPickerView *)pickerView rowHeightForComponent : (NSInteger)component
{
   return defaultCellH;
}

//____________________________________________________________________________________________________
- (NSInteger)pickerView : (UIPickerView *)pickerView numberOfRowsInComponent : (NSInteger)component
{
   return kTotalNumOfTypes;
}

//____________________________________________________________________________________________________
- (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView
{
   return 1;
}

//____________________________________________________________________________________________________
- (UIView *)pickerView : (UIPickerView *)pickerView viewForRow : (NSInteger)row forComponent : (NSInteger)component reusingView : (UIView *)view
{
   UILabel *label = [[UILabel alloc] initWithFrame : CGRectMake(0.f, 0.f, defaultCellW, defaultCellH)];
   label.text = errorTypesStrings[row];
   label.font = [UIFont fontWithName : @"TimesNewRomanPS-BoldMT" size : 14.f];

#ifdef __IPHONE_6_0
   label.textAlignment = NSTextAlignmentCenter;
#else
   label.textAlignment = UITextAlignmentCenter;
#endif

   label.backgroundColor = [UIColor colorWithPatternImage : [UIImage imageNamed : @"text_cell_bkn.png"]];

   return label;
}

//____________________________________________________________________________________________________
- (void)pickerView : (UIPickerView *)thePickerView didSelectRow : (NSInteger)row inComponent : (NSInteger)component
{
   if (row >= 0) {
      [controller setErrorOption : histErrorTypes[row]];
      [controller objectWasModifiedUpdateSelection : YES];
   }
}

#pragma mark ObjectInspectorComponent protocol.
//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = static_cast<TH1*>(o);
   //Read error type from hist.
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
}

@end
