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
- (instancetype) initWithNibName : (NSString *) nibNameOrNil bundle : (NSBundle *) nibBundleOrNil
{
   self = [super initWithNibName : nibNameOrNil bundle : nibBundleOrNil];
   if (self) {
      //Force a view load.
      [self view];
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

   object = static_cast<TVirtualPad *>(o);
   //Result of cast is not checked here, it's done at the upper level.
   logX.on = object->GetLogx();
   logY.on = object->GetLogy();
   logZ.on = object->GetLogz();
}

//____________________________________________________________________________________________________
- (IBAction) logActivated : (UISwitch *) log
{
   assert(object != nullptr && "logActivated:, object is null");

   if (log == logX)
      object->SetLogx(log.on);
   if (log == logY)
      object->SetLogy(log.on);
   if (log == logZ)
      object->SetLogz(log.on);
   
   [controller objectWasModifiedUpdateSelection : YES];//Now picture changed, so picking buffer is invalid.
}

@end
