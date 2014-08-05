#import "ROOTObjectController.h"
#import "HorizontalPickerView.h"
#import "FilledAreaInspector.h"
#import "PatternCell.h"
#import "Constants.h"
#import "ColorCell.h"

//C++ (ROOT) imports:
#import "IOSFillPatterns.h"
#import "TAttFill.h"
#import "TObject.h"

//It's mm file == C++, consts have internal linkage.
const CGFloat defaultCellW = 50.f;
const CGFloat defaultCellH = 50.f;

@implementation FilledAreaInspector  {
   HorizontalPickerView *colorPicker;
   HorizontalPickerView *patternPicker;

   NSMutableArray *colorCells;
   NSMutableArray *patternCells;

   TAttFill *filledObject;

   __weak ROOTObjectController *parentController;
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   using namespace ROOT::iOS::Browser;

   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];

   [self view];

   if (self) {
      const CGRect cellRect = CGRectMake(0.f, 0.f, defaultCellW, defaultCellH);

      colorCells = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
         ColorCell * newCell = [[ColorCell alloc] initWithFrame : cellRect];
         [newCell setRGB : predefinedFillColors[i]];
         [colorCells addObject : newCell];
      }

      colorPicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 15.f, 220.f, 70.f)];
      [colorPicker addItems : colorCells];
      [self.view addSubview : colorPicker];
      colorPicker.pickerDelegate = self;

      patternCells = [[NSMutableArray alloc] init];
      PatternCell *solidFill = [[PatternCell alloc] initWithFrame : cellRect andPattern : 0];
      [solidFill setAsSolid];
      [patternCells addObject : solidFill];

      for (unsigned i = 0; i < ROOT::iOS::GraphicUtils::kPredefinedFillPatterns; ++i) {
         PatternCell *newCell = [[PatternCell alloc] initWithFrame : cellRect andPattern : i];
         [patternCells addObject : newCell];
      }

      patternPicker = [[HorizontalPickerView alloc] initWithFrame:CGRectMake(15.f, 90.f, 220.f, 70.f)];
      [patternPicker addItems : patternCells];
      [self.view addSubview : patternPicker];
      patternPicker.pickerDelegate = self;
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)didReceiveMemoryWarning
{
   [super didReceiveMemoryWarning];
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *) p
{
   parentController = p;
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)obj
{
   //ROOT's standard color pick has 16 colors,
   //I have 16 rows in a color picker.
   //Fill color is some integer index, not from [0 16),
   //but some hardcoded constant (as usually :( ) -
   //see TGColorSelect or something like this.
   //I hold this indices in colorIndices array of constants,
   //since ROOT does not define them.
   //If the object color is one of 16 standard colors,
   //I find the correct row in a picker and rotate picker
   //to this row. If not - it's on zero.
   using namespace ROOT::iOS::Browser;

   //I do not check the result of dynamic_cast here. This is done at upper level.
   filledObject = dynamic_cast<TAttFill *>(obj);

   //Set the row in color picker, using fill color from object.
   const Color_t colorIndex = filledObject->GetFillColor();
   unsigned pickerItem = 0;
   for (unsigned i = 0; i < nROOTDefaultColors; ++i) {
      if (colorIndex == colorIndices[i]) {
         pickerItem = i;
         break;
      }
   }

   [colorPicker setSelectedItem : pickerItem];

   //Look for a fill pattern.
   namespace Fill = ROOT::iOS::GraphicUtils;

   const Style_t fillStyle = filledObject->GetFillStyle();
   if (fillStyle == Fill::solidFillStyle)//I'm sorry, again, hardcoded constant, ROOT does not define it :(.
      pickerItem = 0;
   else
      pickerItem = filledObject->GetFillStyle() % Fill::stippleBase;

   [patternPicker setSelectedItem : pickerItem];
}

//____________________________________________________________________________________________________
- (NSString *) getComponentName
{
   return @"Fill attributes";
}

//____________________________________________________________________________________________________
- (void) setNewColor : (NSInteger) cellIndex
{
   using namespace ROOT::iOS::Browser;

   if (filledObject && parentController) {
      const bool wasHollow = filledObject->GetFillColor() == 0;
      if (cellIndex >= 0 && cellIndex < nROOTDefaultColors) {
         const bool isHollow = colorIndices[cellIndex] == 0;
         filledObject->SetFillColor(colorIndices[cellIndex]);

         if (wasHollow != isHollow)
            [parentController objectWasModifiedUpdateSelection : YES];
         else
            [parentController objectWasModifiedUpdateSelection : NO];
      }
   }
}

//____________________________________________________________________________________________________
- (void) setNewPattern : (NSInteger) cellIndex
{
   namespace Fill = ROOT::iOS::GraphicUtils;

   if (filledObject && parentController) {
      if (cellIndex > 0 && cellIndex <= Fill::kPredefinedFillPatterns) {
         filledObject->SetFillStyle(Fill::stippleBase + cellIndex);
      } else if (!cellIndex) {
         filledObject->SetFillStyle(Fill::solidFillStyle);
      }

      [parentController objectWasModifiedUpdateSelection : NO];
   }
}

#pragma mark - View lifecycle

//____________________________________________________________________________________________________
- (void)viewDidLoad
{
   [super viewDidLoad];
}

//____________________________________________________________________________________________________
- (void)viewDidUnload
{
   [super viewDidUnload];
}

//____________________________________________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation
{
   // Return YES for supported orientations
   return YES;
}

#pragma mark - Color/pattern picker's delegate.

//____________________________________________________________________________________________________
- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *)picker
{
   if (picker == colorPicker) {
      [self setNewColor : item];
   } else {
      [self setNewPattern : item];
   }
}

@end
