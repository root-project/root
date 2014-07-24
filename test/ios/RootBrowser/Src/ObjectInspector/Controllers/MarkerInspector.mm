#import <cassert>

#import "HorizontalPickerView.h"
#import "ObjectViewController.h"
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

EMarkerStyle const markerStyles[] = {kDot, kPlus, kStar, kCircle, kMultiply,
                                     kFullDotSmall, kFullDotMedium, kFullDotLarge,
                                     kFullCircle, kFullSquare, kFullTriangleUp,
                                     kFullTriangleDown, kOpenCircle, kOpenSquare,
                                     kOpenTriangleUp, kOpenDiamond, kOpenCross,
                                     kFullStar, kOpenStar, kOpenTriangleDown,
                                     kFullDiamond, kFullCross};

const unsigned nMarkers = sizeof markerStyles / sizeof markerStyles[0];

//____________________________________________________________________________________________________
bool CanScaleMarker(Style_t style)
{
   return !(style == kDot || style == kFullDotSmall || style == kFullDotMedium);
}

}

@implementation MarkerInspector {
   __weak IBOutlet UIButton *plusBtn;
   __weak IBOutlet UIButton *minusBtn;
   __weak IBOutlet UILabel *sizeLabel;

   HorizontalPickerView *markerStylePicker;
   HorizontalPickerView *markerColorPicker;

   NSMutableArray *styleCells;
   NSMutableArray *colorCells;
   
   __weak ObjectViewController *controller;
   TAttMarker *object;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   using namespace ROOT::iOS::Browser;

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   
   if (self) {
      //Force views load.
      [self view];
      
      styleCells = [[NSMutableArray alloc] init];//]WithCapacity : nMarkers];
      for (unsigned i = 0; i < nMarkers; ++i) {
         MarkerStyleCell * const newCell = [[MarkerStyleCell alloc] initWithFrame : cellRect andMarkerStyle : markerStyles[i]];
         [styleCells addObject : newCell];
      }
      
      markerStylePicker = [[HorizontalPickerView alloc] initWithFrame : CGRectMake(15.f, 15.f, 220.f, 70.f)];
      [markerStylePicker addItems : styleCells];
      [self.view addSubview : markerStylePicker];
      markerStylePicker.pickerDelegate = self;
      
      colorCells = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
         ColorCell * const newCell = [[ColorCell alloc] initWithFrame : cellRect];
         [newCell setRGB : predefinedFillColors[i]];
         [colorCells addObject : newCell];
      }
      
      markerColorPicker = [[HorizontalPickerView alloc] initWithFrame : CGRectMake(15.f, 110.f, 220.f, 70.f)];
      [markerColorPicker addItems : colorCells];
      [self.view addSubview : markerColorPicker];
      markerColorPicker.pickerDelegate = self;
   }

   return self;
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)
   return YES;
}

#pragma mark - Horizontal picker delegate.

//____________________________________________________________________________________________________
- (void) item : (unsigned int) item wasSelectedInPicker : (HorizontalPickerView *) picker
{
   assert(picker != nil && "item:wasSelectedInPicker:, parameter 'picker' is nil");
   assert(object != nullptr && "item:wasSelectedInPicker:, object is null");

   if (picker == markerColorPicker) {
      const unsigned colorIndex = ROOT::iOS::Browser::colorIndices[item];
      object->SetMarkerColor(colorIndex);
      [controller objectWasModifiedUpdateSelection : YES];
   } else if (picker == markerStylePicker) {
      if (item < nMarkers) {
         EMarkerStyle style = markerStyles[item];
         if (CanScaleMarker(style)) {
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
- (void) setObjectController : (ObjectViewController *) c
{
   assert(c != nil && "setObjectController:, parameter 'c' is nil");

   controller = c;
}

//____________________________________________________________________________________________________
- (void) setObject : (TObject *) o
{
   using namespace ROOT::iOS::Browser;
   
   assert(o != nullptr && "setObject:, parameter 'o' is null");

   object = dynamic_cast<TAttMarker *>(o);
   //The result is tested one level up, no check here.

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

   if (!CanScaleMarker(object->GetMarkerStyle())) {
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

#pragma mark - UI interactions.

//____________________________________________________________________________________________________
- (IBAction) plusPressed
{
   assert(object != nullptr && "plusPressed, object is null");

   if (object->GetMarkerSize() + sizeStep > maxMarkerSize)
      return;

   object->SetMarkerSize(object->GetMarkerSize() + sizeStep);
   sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];   
   [controller objectWasModifiedUpdateSelection : YES];
}

//____________________________________________________________________________________________________
- (IBAction) minusPressed
{
   assert(object != nullptr && "minusPressed, object is null");

   if (object->GetMarkerSize() - sizeStep < 1.)
      return;
   
   object->SetMarkerSize(object->GetMarkerSize() - sizeStep);
   sizeLabel.text = [NSString stringWithFormat : @"%.2g", object->GetMarkerSize()];   
   [controller objectWasModifiedUpdateSelection : YES];
}

@end
