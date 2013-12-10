#import <cassert>

#import "ObjectViewController.h"
#import "AxisFontInspector.h"

//C++ (ROOT) imports.
#import "TObject.h"
#import "TAxis.h"

namespace {

const CGFloat defaultCellW = 180.f;
const CGFloat defaultCellH = 44.f;

NSString * const fixedFonts[] =  //These are the strings with font names to use with UILabel.
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

NSString * const fixedFontNames[] = //these are the strings to show in a picker view.
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
   __weak IBOutlet UILabel *titleLabel;
   __weak IBOutlet UIPickerView *fontPicker;

   BOOL isTitleFont;
   __weak ObjectViewController *controller;
   TAxis *object;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibName isTitle : (BOOL) isTitle
{
   if (self = [super initWithNibName : nibName bundle : nil]) {
      [self view];
      // Custom initialization
      isTitleFont = isTitle;
      
      if (isTitleFont)
         titleLabel.text = @"Title font:";
      else
         titleLabel.text = @"Label font:";
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

#pragma mark - ObjectInspectorComponent.

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

   object = dynamic_cast<TAxis *>(o);
   //The result of cast is checked one level up.
   Font_t fontIndex = 0;

   if (isTitleFont)
      fontIndex = object->GetTitleFont() / 10 - 1;
   else
      fontIndex = object->GetLabelFont() / 10 - 1;
   
   if (fontIndex < 0 || fontIndex > nFixedFonts)
      fontIndex = 0;
   
   [fontPicker selectRow : fontIndex inComponent : 0 animated : NO];
}

#pragma mark - font name picker's dataSource.
//____________________________________________________________________________________________________
- (CGFloat) pickerView : (UIPickerView *) pickerView widthForComponent : (NSInteger) component
{
#pragma unused(pickerView, component)
   return defaultCellW;
}

//____________________________________________________________________________________________________
- (CGFloat) pickerView : (UIPickerView *) pickerView rowHeightForComponent : (NSInteger) component
{
#pragma unused(pickerView, component)
   return defaultCellH;
}

//____________________________________________________________________________________________________
- (NSInteger) pickerView : (UIPickerView *) pickerView numberOfRowsInComponent : (NSInteger) component
{
#pragma unused(pickerView, component)
   return nFixedFonts;
}

//____________________________________________________________________________________________________
- (NSInteger) numberOfComponentsInPickerView : (UIPickerView *) pickerView
{
#pragma unused(pickerView)
	return 1;
}

#pragma mark font name picker's delegate.

//____________________________________________________________________________________________________
- (UIView *) pickerView : (UIPickerView *) pickerView viewForRow : (NSInteger)row forComponent : (NSInteger) component reusingView : (UIView *) view
{
#pragma unused(pickerView, component, view)

   UILabel * const label = [[UILabel alloc] initWithFrame : CGRectMake(0.f, 0.f, defaultCellW, defaultCellH)];
   label.text = fixedFontNames[row];
   label.font = [UIFont fontWithName : fixedFonts[row] size : 14.f];
   label.textAlignment = NSTextAlignmentCenter;
   label.backgroundColor = [UIColor colorWithPatternImage : [UIImage imageNamed : @"text_cell_bkn.png"]];
   
   return label;
}

//____________________________________________________________________________________________________
- (void) pickerView : (UIPickerView *) pickerView didSelectRow : (NSInteger) row inComponent : (NSInteger) component
{
#pragma unused(pickerView, component)

   assert(object != nullptr && "pickerView:didSelectRow:component:, object is null");

   const Font_t fontIndex = (row + 1) * 10;
   if (isTitleFont)
      object->SetTitleFont(fontIndex);
   else
      object->SetLabelFont(fontIndex);

   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (void) back
{
   [self.navigationController popViewControllerAnimated : YES];
}

@end
