#import "ROOTObjectController.h"
#import "AxisTitleInspector.h"
#import "AxisColorInspector.h"
#import "AxisFontInspector.h"

//C++ (ROOT) imports.
#import "TObject.h"
#import "TAxis.h"

static const float minTitleOffset = 0.1f;
static const float maxTitleOffset = 10.f;
static const float titleOffsetStep = 0.01f;

static const float minTitleSize = 0.01f;
static const float maxTitleSize = 1.f;
static const float titleSizeStep = 0.01f;

static const float totalHeight = 400.f;
static const float tabBarHeight = 49.f;

@implementation AxisTitleInspector

@synthesize titleField;
@synthesize centered;
@synthesize rotated;
@synthesize offsetLabel;
@synthesize plusOffsetBtn;
@synthesize minusOffsetBtn;
@synthesize sizeLabel;
@synthesize plusSizeBtn;
@synthesize minusSizeBtn;


//____________________________________________________________________________________________________
+ (CGRect) inspectorFrame
{
   return CGRectMake(0.f, tabBarHeight, 250.f, totalHeight - tabBarHeight);
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{

   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self)
      [self view];

   return self;
}

//____________________________________________________________________________________________________
- (void) dealloc
{
   self.titleField = nil;
   self.centered = nil;
   self.rotated = nil;
   self.offsetLabel = nil;
   self.plusOffsetBtn = nil;
   self.minusOffsetBtn = nil;
   
   self.sizeLabel = nil;
   self.plusSizeBtn = nil;
   self.minusSizeBtn = nil;

   [super dealloc];
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
   object = dynamic_cast<TAxis *>(o);

   const char *axisTitle = object->GetTitle();
   if (!axisTitle || !*axisTitle)
      titleField.text = @"";
   else
      titleField.text = [NSString stringWithFormat : @"%s", axisTitle];
      
   centered.on = object->GetCenterTitle();
   rotated.on = object->GetRotateTitle();
   
   offset = object->GetTitleOffset();
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   
   titleSize = object->GetTitleSize();
   if (titleSize > maxTitleSize || titleSize < minTitleSize) {//this is baaad
      titleSize = minTitleSize;
      object->SetTitleSize(titleSize);
      [controller objectWasModifiedUpdateSelection : NO];
   }

   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
}

//____________________________________________________________________________________________________
- (void) back
{
   [self.navigationController popViewControllerAnimated : YES];   
}

//____________________________________________________________________________________________________
- (IBAction) showTitleFontInspector
{
   AxisFontInspector *fontInspector = [[AxisFontInspector alloc] initWithNibName : @"AxisFontInspector" mode : ROOT_IOSObjectInspector::afimTitleFont];

   [fontInspector setROOTObjectController : controller];
   [fontInspector setROOTObject : object];
   
   [self.navigationController pushViewController : fontInspector animated : YES];
   [fontInspector release];
}

//____________________________________________________________________________________________________
- (IBAction) showTitleColorInspector
{
   AxisColorInspector *colorInspector = [[AxisColorInspector alloc] initWithNibName : @"AxisColorInspector" bundle : nil mode : ROOT_IOSObjectInspector::acimTitleColor];

   [colorInspector setROOTObjectController : controller];
   [colorInspector setROOTObject : object];
   
   [self.navigationController pushViewController : colorInspector animated : YES];
   [colorInspector release];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldDidEndOnExit : (id) sender
{
   object->SetTitle([titleField.text cStringUsingEncoding : [NSString defaultCStringEncoding]]);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) textFieldEditingDidEnd : (id) sender
{
   [sender resignFirstResponder];
}

//____________________________________________________________________________________________________
- (IBAction) centerTitle
{
   object->CenterTitle(centered.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) rotateTitle
{
   object->RotateTitle(rotated.on);
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) plusOffset
{
   if (offset + titleOffsetStep > maxTitleOffset)
      return;
   
   offset += titleOffsetStep;
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   object->SetTitleOffset(offset);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusOffset
{
   if (offset - titleOffsetStep < minTitleOffset)
      return;
   
   offset -= titleOffsetStep;
   offsetLabel.text = [NSString stringWithFormat:@"%.2f", offset];
   object->SetTitleOffset(offset);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) plusSize
{
   if (titleSize + titleSizeStep > maxTitleSize)
      return;
   
   titleSize += titleSizeStep;
   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
   object->SetTitleSize(titleSize);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) minusSize
{
   if (titleSize - titleSizeStep < minTitleSize)
      return;
   
   titleSize -= titleSizeStep;
   sizeLabel.text = [NSString stringWithFormat:@"%.2f", titleSize];
   object->SetTitleSize(titleSize);
   
   [controller objectWasModifiedUpdateSelection : NO];
}


@end
