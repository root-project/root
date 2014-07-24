#import <cassert>

#import "PadOptionsController.h"
#import "PatternCell.h"
#import "PadView.h"

//C++ code (ROOT)
#import "IOSFillPatterns.h"
#import "IOSPad.h"

const CGFloat defaultCellWidth = 80.f;
const CGFloat defaultCellHeight = 44.f;
const CGRect defaultCellFrame = CGRectMake(0.f, 0.f, defaultCellWidth, defaultCellHeight);

const NSInteger nPredefinedColors = 16;
const CGFloat predefinedFillColors[nPredefinedColors][3] = {
{1., 1., 1.},
{0., 0., 0.},
{251 / 255., 0., 24 / 255.},
{40 / 255., 253 / 255., 44 / 255.},
{31 / 255., 29 / 255., 251 / 255.},
{253 / 255., 254 / 255., 52 / 255.},
{253 / 255., 29 / 255., 252 / 255.},
{53 / 255., 1., 254 / 255.},
{94 / 255., 211 / 255., 90 / 255.},
{92 / 255., 87 / 255., 214 / 255.},
{135 / 255., 194 / 255., 164 / 255.},
{127 / 255., 154 / 255., 207 / 255.},
{211 / 255., 206 / 255., 138 / 255.},
{220 / 255., 185 / 255., 138 / 255.},
{209 / 255., 89 / 255., 86 / 255.},
{147 / 255., 29 / 255., 251 / 255.}
};


//Color indices in a standard ROOT's color selection control:
const unsigned colorIndices[16] = {
0, 1, 2, 3,
4, 5, 6, 7,
8, 9, 30, 38,
41, 42, 50, 51};


@implementation PadOptionsController {
   //UI:
   __weak IBOutlet UISwitch *tickX_;
   __weak IBOutlet UISwitch *tickY_;

   __weak IBOutlet UISwitch *gridX_;
   __weak IBOutlet UISwitch *gridY_;

   __weak IBOutlet UISwitch *logX_;
   __weak IBOutlet UISwitch *logY_;
   __weak IBOutlet UISwitch *logZ_;
   
   __weak IBOutlet UIPickerView *colorPicker_;
   __weak IBOutlet UIPickerView *patternPicker_;

   //Controlled objects:
   ROOT::iOS::Pad *pad;
   PadView *padView;
}

#pragma mark - View lifecycle

//_________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

    return YES;
}

#pragma mark - editing.

//_________________________________________________________________
- (void) setView : (PadView *) view andPad : (ROOT::iOS::Pad *) newPad
{
   assert(view != nil && "setView:andPad:, parameter 'view' is nil");
   assert(newPad != nullptr && "setView:andPad:, parameter 'newPad' is null");

   padView = view;
   pad = newPad;
   
   gridX_.on = pad->GetGridx();
   gridY_.on = pad->GetGridy();
   tickX_.on = pad->GetTickx();
   tickY_.on = pad->GetTicky();
   
   logX_.on = pad->GetLogx();
   logY_.on = pad->GetLogy();
   logZ_.on = pad->GetLogz();
}

//_________________________________________________________________
- (IBAction) tickActivated : (id) control
{
   assert([control isKindOfClass : [UISwitch class]] &&
          "tickActivated:, parameter 'control' has a wrong type");

   const unsigned on = [(UISwitch *)control isOn];
   if (control == tickX_)
      pad->SetTickx(on);
   else if (control == tickY_)
      pad->SetTicky(on);
   //else assert.
   
   [padView setNeedsDisplay];
}

//_________________________________________________________________
- (IBAction) gridActivated : (id) control
{
   assert([control isKindOfClass : [UISwitch class]] &&
          "gridActivated:, parameter 'control' has a wrong type");

   const unsigned on = [(UISwitch *)control isOn];
   if (control == gridX_)
      pad->SetGridx(on);
   else if (control == gridY_)
      pad->SetGridy(on);
   //else assert.

   [padView setNeedsDisplay];
}

//_________________________________________________________________
- (IBAction) logActivated : (id) control
{
   assert([control isKindOfClass : [UISwitch class]] &&
          "logActivated:, parameter 'control' has a wrong type");

   const unsigned on = [(UISwitch *)control isOn];
   if (control == logX_)
      pad->SetLogx(on);
   if (control == logY_)
      pad->SetLogy(on);
   if (control == logZ_)
      pad->SetLogz(on);


   //Else of all ifs must be an assert.

   [padView setNeedsDisplay];
}

#pragma mark - color/pattern picker dataSource.
//_________________________________________________________________
- (CGFloat) pickerView : (UIPickerView *) pickerView widthForComponent : (NSInteger) component
{
#pragma unused(pickerView, component)

   return defaultCellWidth;
}

//_________________________________________________________________
- (CGFloat) pickerView : (UIPickerView *) pickerView rowHeightForComponent : (NSInteger) component
{
#pragma unused(pickerView, component)

   return defaultCellHeight;
}

//_________________________________________________________________
- (NSInteger) pickerView : (UIPickerView *) pickerView numberOfRowsInComponent : (NSInteger) component
{
#pragma unused(component)

   if (pickerView == colorPicker_)
      return nPredefinedColors;
   else if (pickerView == patternPicker_)
      return ROOT::iOS::GraphicUtils::kPredefinedFillPatterns + 1;//+ 1 for a 'solid'.

   assert(0 && "pickerView:numberOfRowsInComponent:, parameter 'pickerView' is invalid");

   return 0;
}

//_________________________________________________________________
- (NSInteger) numberOfComponentsInPickerView : (UIPickerView *) pickerView
{
   //We have two pickers, each with exactly one 'wheel'.
   return 1;
}

#pragma mark UIPickerViewDelegate

// tell the picker which view to use for a given component and row, we have an array of views to show
//_________________________________________________________________
- (UIView *) pickerView : (UIPickerView *) pickerView viewForRow : (NSInteger) row
           forComponent : (NSInteger) component reusingView : (UIView *) view
{
#pragma unused(component)

   if (pickerView == colorPicker_) {
      assert(row < nPredefinedColors && row >= 0 &&
             "pickerView:viewForRow:forComponent:reusingView:, row is out of bounds");
      const CGFloat * const rgb = predefinedFillColors[row];
      UIView * newCell = view;
      if (!newCell)
         newCell = [[UIView alloc] initWithFrame:defaultCellFrame];
      newCell.backgroundColor = [UIColor colorWithRed : rgb[0] green : rgb[1] blue : rgb[2] alpha : 1.f];

      return newCell;
   } else if (pickerView == patternPicker_) {
      //Row 0 is a special case - I have to call the setAsSolid method.
      assert(row >= 0 && row <= ROOT::iOS::GraphicUtils::kPredefinedFillPatterns &&
             "pickerView:viewForRow:forCoponent:reusingView:, row is out of bounds");//<= -> +1 for 'solid'.
      
      if (view && [view isKindOfClass : [PatternCell class]]) {
         PatternCell * const reuseCell = (PatternCell *)view;
         if (!row)
            [reuseCell setAsSolid];
         else
            [reuseCell setPattern : row - 1];

         return reuseCell;
      } else {
         PatternCell * const newCell = [[PatternCell alloc] initWithFrame : defaultCellFrame andPattern : row ? row - 1 : 0];
         if (!row)
            [newCell setAsSolid];
         return newCell;
      }
   }

   assert(0 && "pickerView:viewForRow:forCoponent:reusingView:, parameter 'pickerView' is invalid");

   return nil;
}

//_________________________________________________________________
- (void) pickerView : (UIPickerView *) pickerView didSelectRow : (NSInteger) row inComponent : (NSInteger) component
{
#pragma unused(component)

   assert(pad != nullptr && "pickerView:didSelectRow:inComponent:, pad is null");
   assert(padView != nil && "pickerView:didSelectRow:inComponent:, padView is nil");

   if (pickerView == colorPicker_) {
      if (row >= 0 && row < 16) {
         pad->SetFillColor(colorIndices[row]);
         [padView setNeedsDisplay];
      }
   } else if (pickerView == patternPicker_) {
      //<= because of solid fill pattern.
      if (row > 0 && row <= ROOT::iOS::GraphicUtils::kPredefinedFillPatterns) {
         pad->SetFillStyle(3000 + row);
         [padView setNeedsDisplay];
      } else if (!row) {
         pad->SetFillStyle(1001);
         [padView setNeedsDisplay];
      }
   } //else must be an assert.
}

@end
