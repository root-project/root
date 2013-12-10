#import <cassert>

#import "HorizontalPickerView.h"
#import "ObjectViewController.h"
#import "LineWidthPicker.h"
#import "LineInspector.h"
#import "LineStyleCell.h"
#import "ColorCell.h"
#import "Constants.h"

//C++ (ROOT) imports.
#import "TAttLine.h"
#import "TObject.h"
#import "TGraph.h"

//It's mm file == C++, consts have internal linkage.
const int minLineWidth = 1;
const int maxLineWidth = 15;
const CGRect cellFrame = CGRectMake(0.f, 0.f, 50.f, 50.f);

@implementation LineInspector {
   __weak IBOutlet LineWidthPicker *lineWidthPicker;
   NSMutableArray *lineStyles;
   NSMutableArray *lineColors;

   HorizontalPickerView *lineColorPicker;
   HorizontalPickerView *lineStylePicker;

   int lineWidth;

   __weak ObjectViewController *controller;
   TAttLine *object;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   using namespace ROOT::iOS::Browser;

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self) {
      //Force views load.
      [self view];
      //Array with cells for "Line style" picker.
      lineStyles = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < 10; ++i) {
         LineStyleCell *newCell = [[LineStyleCell alloc] initWithFrame : cellFrame lineStyle : i + 1];
         [lineStyles addObject : newCell];
      }
      
      lineStylePicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 20.f, 220.f, 70.f)];
      [lineStylePicker addItems : lineStyles];
      [self.view addSubview : lineStylePicker];
      
      lineStylePicker.pickerDelegate = self;
      
      lineColors = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
         ColorCell *newCell = [[ColorCell alloc] initWithFrame : cellFrame];
         [newCell setRGB : predefinedFillColors[i]];
         [lineColors addObject : newCell];
      }

      lineColorPicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 105, 220.f, 70.f)];
      [lineColorPicker addItems : lineColors];
      [self.view addSubview : lineColorPicker];
      
      lineColorPicker.pickerDelegate = self;
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

   using namespace ROOT::iOS::Browser;

   object = dynamic_cast<TAttLine *>(o);
   assert("setObject:, object has a wrong type, TAttLine expected");
   
   unsigned item = 0;
   const Style_t lineStyle = object->GetLineStyle();
   if (lineStyle >= 1 && lineStyle <= 10)
      item = lineStyle - 1;

   [lineStylePicker setSelectedItem : item];

   item = 1;//black.
   const Color_t colorIndex = object->GetLineColor();
   for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
      if (colorIndex == colorIndices[i]) {
         item = i;
         break;
      }
   }
   
   [lineColorPicker setSelectedItem : item];
   
   //Line width is expected to be line width in pixels,
   //but it can hold additional information in case of
   //TGraph and have value like -2014.
   lineWidth = object->GetLineWidth();
   if (lineWidth < minLineWidth || lineWidth > maxLineWidth) {
      if (dynamic_cast<TGraph *>(o)) {
         //"Formula" from ROOT.
         lineWidth = TMath::Abs(lineWidth) % 100;
         //Still, line width can be out of [1,15] range!
         if (!lineWidth)
            lineWidth = 1;
         else if (lineWidth > maxLineWidth)
            lineWidth = maxLineWidth;
      } else
         lineWidth = minLineWidth;
   }

   [lineWidthPicker setLineWidth : lineWidth];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Line attributes";
}

#pragma mark - Horizontal picker delegate.

//____________________________________________________________________________________________________
- (void) item : (unsigned int) item wasSelectedInPicker : (HorizontalPickerView *) picker
{
   assert(picker != nil && "item:wasSelectedInPicker:, parameter 'picker' is nil");
   assert(object != nullptr && "item:wasSelectedInPicker:, object is null");

   using namespace ROOT::iOS::Browser;

   if (picker == lineColorPicker) {
      assert(item < nROOTDefaultColors && "item:wasSelectedInPicker:, parameter 'item' is out of bounds");
      const unsigned colorIndex = colorIndices[item];
      object->SetLineColor(colorIndex);
   } else {
      //why 10 is hardcoded?
      assert(item < 10 && "item:wasSelectedInPicker:, parameter 'item' is out of bounds");
      object->SetLineStyle(item + 1);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

#pragma mark - Code to deal with line width's insanity

//____________________________________________________________________________________________________
- (void) updateROOTLineWidth
{
   assert(object != nullptr && "updateROOTLineWidth, object is null");

   if (dynamic_cast<TGraph *>(object)) {
      const int fakeLineWidth = int(object->GetLineWidth()) / 100 * 100;
      if (fakeLineWidth >= 0)
         object->SetLineWidth(fakeLineWidth + lineWidth);
      else
         object->SetLineWidth(-(TMath::Abs(fakeLineWidth) + lineWidth));
   } else
      object->SetLineWidth(lineWidth);
}

#pragma mark - Button's handlers.



//____________________________________________________________________________________________________
- (IBAction) decLineWidth
{
   if (lineWidth == minLineWidth)
      return;

   --lineWidth;
   [lineWidthPicker setLineWidth : lineWidth];
   
   [self updateROOTLineWidth];

   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) incLineWidth
{
   if (lineWidth == maxLineWidth)
      return;
      
   ++lineWidth;
   [lineWidthPicker setLineWidth : lineWidth];
   
   [self updateROOTLineWidth];

   [controller objectWasModifiedUpdateSelection : NO];
}

@end
