#import "ROOTObjectController.h"
#import "AxisFontInspector.h"

//C++ (ROOT) imports.
#import "TObject.h"
#import "TAxis.h"

namespace {

const CGFloat defaultCellW = 180.f;
const CGFloat defaultCellH = 44.f;

NSString *fixedFonts[] =  //These are the strings with font names to use with UILabel.
                                     {
                                      @"TimesNewRomanPS-ItalicMT",
                                      @"TimesNewRomanPS-BoldMT",
                                      @"TimesNewRomanPS-BoldItalicMT",
                                      @"Helvetica",
                                      @"Helvetica-Oblique",
                                      @"Helvetica-Bold",
                                      @"Helvetica-BoldOblique",
                                      @"Courier",
                                      @"Courier-Oblique",
                                      @"Courier-Bold",
                                      @"Courier-BoldOblique",
                                      @"symbol",//No custom fonts yet.
                                      @"TimesNewRomanPSMT"
                                     };

NSString *fixedFontNames[] = //these are the strings to show in a picker view.
                                     {
                                      @"Times New Roman",
                                      @"Times New Roman",
                                      @"Times New Roman",
                                      @"Helvetica",
                                      @"Helvetica",
                                      @"Helvetica",
                                      @"Helvetica",
                                      @"Courier",
                                      @"Courier",
                                      @"Courier",
                                      @"Courier",
                                      @"Symbol",//No custom fonts yet.
                                      @"Times New Roman"
                                     };
                                     
const unsigned nFixedFonts = sizeof fixedFonts / sizeof fixedFonts[0];

}

@implementation AxisFontInspector {
   ROOT_IOSObjectInspector::AxisFontInspectorMode mode;
   __weak ROOTObjectController *controller;
   TAxis *object;
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibName mode : (ROOT_IOSObjectInspector::AxisFontInspectorMode)m
{
   using namespace ROOT_IOSObjectInspector;

   self = [super initWithNibName : nibName bundle : nil];
   
   [self view];
   
   if (self) {
      // Custom initialization
      mode = m;
      if (mode == afimTitleFont)
         titleLabel.text = @"Title font:";
      else if (mode == afimLabelFont)
         titleLabel.text = @"Label font:";
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
- (BOOL)shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation)interfaceOrientation
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
   using namespace ROOT_IOSObjectInspector;

   object = dynamic_cast<TAxis *>(o);
   Font_t fontIndex = 0;

   if (mode == afimTitleFont)
      fontIndex = object->GetTitleFont() / 10 - 1;
   else if (mode == afimLabelFont)
      fontIndex = object->GetLabelFont() / 10 - 1;
   
   if (fontIndex < 0 || fontIndex > nFixedFonts)
      fontIndex = 0;
   
   [fontPicker selectRow : fontIndex inComponent : 0 animated : NO];
}

#pragma mark - color/pattern picker's dataSource.
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
   return nFixedFonts;
}

//____________________________________________________________________________________________________
- (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView
{
	return 1;
}

#pragma mark color/pattern picker's delegate.

//____________________________________________________________________________________________________
- (UIView *) pickerView : (UIPickerView *)pickerView viewForRow : (NSInteger)row forComponent : (NSInteger)component reusingView : (UIView *)view
{
   UILabel *label = [[UILabel alloc] initWithFrame : CGRectMake(0.f, 0.f, defaultCellW, defaultCellH)];
   label.text = fixedFontNames[row];
   label.font = [UIFont fontWithName : fixedFonts[row] size : 14.f];
   label.textAlignment = UITextAlignmentCenter;
   label.backgroundColor = [UIColor colorWithPatternImage : [UIImage imageNamed : @"text_cell_bkn.png"]];
   
   return label;
}

//____________________________________________________________________________________________________
- (void) pickerView : (UIPickerView *)thePickerView didSelectRow : (NSInteger)row inComponent : (NSInteger)component
{
   using namespace ROOT_IOSObjectInspector;

   const Font_t fontIndex = (row + 1) * 10;
   if (mode == afimTitleFont)
      object->SetTitleFont(fontIndex);
   else if (mode == afimLabelFont)
      object->SetLabelFont(fontIndex);

   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (void) back
{
   [self.navigationController popViewControllerAnimated : YES];
}

@end
