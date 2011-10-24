#import "ROOTObjectController.h"
#import "PadLogScaleInspector.h"

//C++ (ROOT) imports.
#import "TVirtualPad.h"
#import "TObject.h"


@implementation PadLogScaleInspector

@synthesize logX;
@synthesize logY;
@synthesize logZ;

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
    
   [self view];
    
   if (self) {
   }

   return self;
}

//____________________________________________________________________________________________________
- (void)dealloc
{
   self.logX = nil;
   self.logY = nil;
   self.logZ = nil;

   [super dealloc];
}

//____________________________________________________________________________________________________
- (void) setROOTObjectController : (ROOTObjectController *)c
{
   controller = c;
}

//____________________________________________________________________________________________________
- (void) setROOTObject : (TObject *)o
{
   object = static_cast<TVirtualPad *>(o);
   //Result of cast is not checked here, it's done at the upper level.
   logX.on = object->GetLogx();
   logY.on = object->GetLogy();
   logZ.on = object->GetLogz();
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
- (IBAction) logActivated : (UISwitch *) log
{
   if (log == logX)
      object->SetLogx(log.on);
   if (log == logY)
      object->SetLogy(log.on);
   if (log == logZ)
      object->SetLogz(log.on);
   
   [controller objectWasModifiedUpdateSelection : YES];//Now picture changed, so picking buffer is invalid.
}

@end
