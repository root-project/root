#import "HorizontalPickerView.h"
#import "ROOTObjectController.h"
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

@interface LineInspector () {
   NSMutableArray *lineStyles;
   NSMutableArray *lineColors;

   HorizontalPickerView *lineColorPicker;
   HorizontalPickerView *lineStylePicker;

   int lineWidth;

   __weak ROOTObjectController *controller;
   TAttLine *object;
}

@end

@implementation LineInspector

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   using namespace ROOT_IOSBrowser;

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self) {
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
   using namespace ROOT_IOSBrowser;

   object = dynamic_cast<TAttLine *>(o);
   
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
- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *)picker
{
   if (picker == lineColorPicker) {
      if (item < ROOT_IOSBrowser::nROOTDefaultColors) {
         const unsigned colorIndex = ROOT_IOSBrowser::colorIndices[item];
         object->SetLineColor(colorIndex);
      } else
         NSLog(@"check the code, bad item index from horizontal picker: %u, must be < %u", item, ROOT_IOSBrowser::nROOTDefaultColors);
   } else {
      if (item < 10)
         object->SetLineStyle(item + 1);
      else
         NSLog(@"check the code, bad item index from horizontal picker: %u must be < 11", item);
   }
   
   [controller objectWasModifiedUpdateSelection : NO];
}

#pragma mark - Code to deal with line width's insanity

//____________________________________________________________________________________________________
- (void) updateROOTLineWidth
{
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
