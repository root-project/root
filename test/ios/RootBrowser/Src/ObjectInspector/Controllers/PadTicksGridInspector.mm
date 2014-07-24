#import <cassert>

#import "PadTicksGridInspector.h"
#import "ObjectViewController.h"

#import "TVirtualPad.h"
#import "TObject.h"

@implementation PadTicksGridInspector {
   __weak IBOutlet UISwitch *gridX;
   __weak IBOutlet UISwitch *gridY;
   __weak IBOutlet UISwitch *ticksX;
   __weak IBOutlet UISwitch *ticksY;

   __weak ObjectViewController *controller;
   TVirtualPad *object;
}

//____________________________________________________________________________________________________
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];

   if (self)
      [self view];
   
   return self;
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

   object = static_cast<TVirtualPad *>(o);
   
   //I do not check the result of cast here, it's done on upper level.
   gridX.on = object->GetGridx();
   gridY.on = object->GetGridy();
   ticksX.on = object->GetTickx();
   ticksY.on = object->GetTicky();
}

#pragma mark - Interface orientation.

//____________________________________________________________________________________________________
- (BOOL) shouldAutorotateToInterfaceOrientation : (UIInterfaceOrientation) interfaceOrientation
{
#pragma unused(interfaceOrientation)

   return YES;
}

#pragma mark - Inspector's actions.

//____________________________________________________________________________________________________
- (IBAction) gridActivated : (UISwitch *) g
{
   assert(object != nullptr && "gridActivated:, object is null");

   if (g == gridX)
      object->SetGridx(g.on);
   else if (g == gridY)
      object->SetGridy(g.on);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

//____________________________________________________________________________________________________
- (IBAction) ticksActivated : (UISwitch *) t
{
   assert(object != nullptr && "ticksActivated:, object is null");

   if (t == ticksX)
      object->SetTickx(t.on);
   else if (t == ticksY)
      object->SetTicky(t.on);
   
   [controller objectWasModifiedUpdateSelection : NO];
}

@end