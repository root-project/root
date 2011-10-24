#import "PadOptionsController.h"
#import "PatternCell.h"
#import "ColorCell.h"

//C++ code (ROOT)
#import "IOSFillPatterns.h"
#import "IOSPad.h"

static const double predefinedFillColors[16][3] = 
{
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
static const unsigned colorIndices[16] = {0, 1, 2, 3,
                                          4, 5, 6, 7,
                                          8, 9, 30, 38,
                                          41, 42, 50, 51};

@implementation PadOptionsController

@synthesize tickX = tickX_;
@synthesize tickY = tickY_;
@synthesize gridX = gridX_;
@synthesize gridY = gridY_;
@synthesize logX = logX_;
@synthesize logY = logY_;
@synthesize logZ = logZ_;
@synthesize colorPicker = colorPicker_;
@synthesize patternPicker = patternPicker_;
@synthesize colors = colors_;
@synthesize patterns = patterns_;

//_________________________________________________________________
- (id)initWithNibName:(NSString *)nibNameOrNil bundle:(NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName:nibNameOrNil bundle:nibBundleOrNil];
   if (self) {
      // Custom initialization
      
      //Color views.
      colors_ = [[NSMutableArray alloc] init];
      for (unsigned i = 0; i < 16; ++i) {
         ColorCell * newCell = [[ColorCell alloc] initWithFrame : CGRectMake(0.f, 0.f, 80.f, 44.f)];
         [newCell setRGB : predefinedFillColors[i]];
         [colors_ addObject : newCell];
         [newCell release];
      }
      
      //Patterns.
      patterns_ = [[NSMutableArray alloc] init];
      
      //The first pattern - solid fill.
      PatternCell *solidFill = [[PatternCell alloc] initWithFrame : CGRectMake(0.f, 0.f, 80.f, 44.f) andPattern : 0];
      [solidFill setAsSolid];
      [patterns_ addObject : solidFill];
      [solidFill release];
      
      for (unsigned i = 0; i < ROOT::iOS::GraphicUtils::kPredefinedFillPatterns; ++i) {
         PatternCell *newCell = [[PatternCell alloc] initWithFrame : CGRectMake(0.f, 0.f, 80.f, 44.f) andPattern : i];
         [patterns_ addObject : newCell];
         [newCell release];
      }
      
      //Pattern views.
   }
   
   return self;
}

//_________________________________________________________________
- (void)dealloc
{
   self.tickX = nil;
   self.tickY = nil;
   
   self.gridX = nil;
   self.gridY = nil;
   
   self.logX = nil;
   self.logY = nil;
   self.logZ = nil;
   
   self.colorPicker = nil;
   self.patternPicker = nil;
   
   [colors_ release];
   [patterns_ release];
   
   [super dealloc];
}

//_________________________________________________________________
- (void)didReceiveMemoryWarning
{
   // Releases the view if it doesn't have a superview.
   [super didReceiveMemoryWarning];
   // Release any cached data, images, etc that aren't in use.
}

#pragma mark - View lifecycle

//_________________________________________________________________
- (void)viewDidLoad
{
   [super viewDidLoad];
   // Do any additional setup after loading the view from its nib.
}

//_________________________________________________________________
- (void)viewDidUnload
{
   [super viewDidUnload];
   // Release any retained subviews of the main view.
   // e.g. self.myOutlet = nil;
}

//_________________________________________________________________
- (BOOL)shouldAutorotateToInterfaceOrientation:(UIInterfaceOrientation)interfaceOrientation
{
    // Return YES for supported orientations
	return YES;
}

#pragma mark - editing.

//_________________________________________________________________
- (void) setView : (PadView *) view andPad : (ROOT::iOS::Pad *) newPad
{
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
   const unsigned on = [control isOn];
   if (control == tickX_) {
      pad->SetTickx(on);
   } else if (control == tickY_) {
      pad->SetTicky(on);
   }
   
   [padView setNeedsDisplay];
}

//_________________________________________________________________
- (IBAction) gridActivated : (id) control
{
   const unsigned on = [control isOn];
   if (control == gridX_) {
      pad->SetGridx(on);
   } else if (control == gridY_) {
      pad->SetGridy(on);
   }
   
   [padView setNeedsDisplay];
}

//_________________________________________________________________
- (IBAction) logActivated : (id) control
{
   const unsigned on = [control isOn];
   
   if (control == logX_)
      pad->SetLogx(on);
   
   if (control == logY_)
      pad->SetLogy(on);
      
   if (control == logZ_)
      pad->SetLogz(on);

   [padView setNeedsDisplay];
}

#pragma mark - color/pattern picker dataSource.
//_________________________________________________________________
- (CGFloat)pickerView:(UIPickerView *)pickerView widthForComponent:(NSInteger)component
{
   return 80.;
}

//_________________________________________________________________
- (CGFloat)pickerView:(UIPickerView *)pickerView rowHeightForComponent:(NSInteger)component
{
   return 44.;
}

//_________________________________________________________________
- (NSInteger)pickerView:(UIPickerView *)pickerView numberOfRowsInComponent:(NSInteger)component
{
   if (pickerView == colorPicker_)
      return [colors_ count];
   else if (pickerView == patternPicker_)
      return [patterns_ count];
   return 0;
}

//_________________________________________________________________
- (NSInteger)numberOfComponentsInPickerView:(UIPickerView *)pickerView
{
	return 1;
}

#pragma mark UIPickerViewDelegate

// tell the picker which view to use for a given component and row, we have an array of views to show
//_________________________________________________________________
- (UIView *)pickerView:(UIPickerView *)pickerView viewForRow:(NSInteger)row
		  forComponent:(NSInteger)component reusingView:(UIView *)view
{
   if (pickerView == colorPicker_)
      return [colors_ objectAtIndex : row];
   else if (pickerView == patternPicker_)
      return [patterns_ objectAtIndex : row];

   return 0;
}

//_________________________________________________________________
- (void)pickerView:(UIPickerView *)thePickerView didSelectRow:(NSInteger)row inComponent:(NSInteger)component {
   
   if (thePickerView == colorPicker_) {
      if (row >= 0 && row < 16) {
         pad->SetFillColor(colorIndices[row]);
         [padView setNeedsDisplay];
      }
   } else if (thePickerView == patternPicker_) {
      //<= because of solid fill pattern.
      if (row > 0 && row <= ROOT::iOS::GraphicUtils::kPredefinedFillPatterns) {
       //  NSLog(@"%p", pad);
         pad->SetFillStyle(3000 + row);
         [padView setNeedsDisplay];
      } else if (!row) {
         pad->SetFillStyle(1001);
         [padView setNeedsDisplay];
      }
   }
}

@end
