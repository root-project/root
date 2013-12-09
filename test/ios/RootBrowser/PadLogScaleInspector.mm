#import <cassert>

#import "ObjectViewController.h"
#import "PadLogScaleInspector.h"

//C++ (ROOT) imports.
#import "TVirtualPad.h"
#import "TObject.h"

@implementation PadLogScaleInspector {
   __weak IBOutlet UISwitch *logX;
   __weak IBOutlet UISwitch *logY;
   __weak IBOutlet UISwitch *logZ;

   __weak ObjectViewController *controller;
   TVirtualPad *object;
}

//____________________________________________________________________________________________________
- (id)initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   [self view];
   return self;
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
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

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
