#import "HorizontalPickerView.h"
#import "ROOTObjectController.h"
#import "MarkerInspector.h"
#import "MarkerStyleCell.h"
#import "ColorCell.h"
#import "Constants.h"

//C++ (ROOT) imports.
#import "TAttMarker.h"
#import "TObject.h"

namespace {

const CGRect cellRect = CGRectMake(0.f, 0.f, 50.f, 50.f);

const CGFloat maxMarkerSize = 5.f;
const CGFloat sizeStep = 0.1f;

EMarkerStyle markerStyles[] = {kDot, kPlus, kStar, kCircle, kMultiply,
                               kFullDotSmall, kFullDotMedium, kFullDotLarge,
                               kFullCircle, kFullSquare, kFullTriangleUp,
                               kFullTriangleDown, kOpenCircle, kOpenSquare,
                               kOpenTriangleUp, kOpenDiamond, kOpenCross,
                               kFullStar, kOpenStar, kOpenTriangleDown,
                               kFullDiamond, kFullCross};

const unsigned nMarkers = sizeof markerStyles / sizeof markerStyles[0];

//____________________________________________________________________________________________________
BOOL canScaleMarker(Style_t style)
{
   if (style == kDot || style == kFullDotSmall || style == kFullDotMedium)
      return NO;
   return YES;
}

}

@implementation MarkerInspector {
   HorizontalPickerView *markerStylePicker;
   HorizontalPickerView *markerColorPicker;

   NSMutableArray *styleCells;
   NSMutableArray *colorCells;
   
   __weak ROOTObjectController *controller;
   TAttMarker *object;
}

//____________________________________________________________________________________________________
- (id) initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   using namespace ROOT::iOS::Browser;

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   
   if (self) {
      [self view];
      
      styleCells = [[NSMutableArray alloc] init];//]WithCapacity : nMarkers];
      for (unsigned i = 0; i < nMarkers; ++i) {
         MarkerStyleCell *newCell = [[MarkerStyleCell alloc] initWithFrame : cellRect andMarkerStyle : markerStyles[i]];
         [styleCells addObject : newCell];
      }
      
      markerStylePicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 15.f, 220.f, 70.f)];
      [markerStylePicker addItems : styleCells];
      [self.view addSubview : markerStylePicker];
      markerStylePicker.pickerDelegate = self;
      
      colorCells = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
         ColorCell *newCell = [[ColorCell alloc] initWithFrame : cellRect];
         [newCell setRGB : predefinedFillColors[i]];
         [colorCells addObject : newCell];
      }
      
      markerColorPicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 110.f, 220.f, 70.f)];
      [markerColorPicker addItems : colorCells];
      [self.view addSubview : markerColorPicker];
      markerColorPicker.pickerDelegate = self;
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
- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *)picker
{
   if (picker == markerColorPicker) {
      const unsigned colorIndex = ROOT::iOS::Browser::colorIndices[item];
      object->SetMarkerColor(colorIndex);
      [controller objectWasModifiedUpdateSelection : YES];
   } else if (picker == markerStylePicker) {
      if (item < nMarkers) {
         EMarkerStyle style = markerStyles[item];
         if (canScaleMarker(style)) {
            plusBtn.enabled = YES;
            minusBtn.enabled = YES;
            sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];
         } else {
            plusBtn.enabled = NO;
            minusBtn.enabled = NO;
            sizeLabel.text = @"1";
         }

         object->SetMarkerStyle(style);
      } else {
         NSLog(@"check horizontal picker code, got item index %u, must be < %u", item, nMarkers);
      }

      [controller objectWasModifiedUpdateSelection : YES];
   }
}

#pragma mark ObjectInspectorComponent protocol.

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   using namespace ROOT::iOS::Browser;

   object = dynamic_cast<TAttMarker *>(o);

   unsigned item = 0;
   const EMarkerStyle style = EMarkerStyle(object->GetMarkerStyle());//Mess with all these styles and EMarkerStyles.
   for (unsigned i = 0; i < nMarkers; ++i) {
      if (style == markerStyles[i]) {
         item = i;
         break;
      }
   }
   
   [markerStylePicker setSelectedItem : item];

   //Extract marker color.
   //The same predefined 16 colors as with fill color.
   item = 1;//?
   const Color_t colorIndex = object->GetMarkerColor();
   
   for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
      if (colorIndex == colorIndices[i]) {
         item = i;
         break;
      }
   }
   
   [markerColorPicker setSelectedItem : item];

   if (!canScaleMarker(object->GetMarkerStyle())) {
      plusBtn.enabled = NO;
      minusBtn.enabled = NO;
      sizeLabel.text = @"1";
   } else {
      plusBtn.enabled = YES;
      minusBtn.enabled = YES;
      sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];
   }
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Marker attributes";
}

//____________________________________________________________________________________________________
- (IBAction) plusPressed
{

   if (object->GetMarkerSize() + sizeStep > maxMarkerSize)
      return;

   object->SetMarkerSize(object->GetMarkerSize() + sizeStep);
   sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];   
   [controller objectWasModifiedUpdateSelection : YES];
}

//____________________________________________________________________________________________________
- (IBAction) minusPressed
{
   if (object->GetMarkerSize() - sizeStep < 1.)
      return;
   
   object->SetMarkerSize(object->GetMarkerSize() - sizeStep);
   sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];   
   [controller objectWasModifiedUpdateSelection : YES];
}

@end
