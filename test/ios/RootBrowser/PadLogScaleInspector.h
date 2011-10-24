#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;
class TVirtualPad;
class TObject;

@interface PadLogScaleInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UISwitch *logX;
   IBOutlet UISwitch *logY;
   IBOutlet UISwitch *logZ;
   
   ROOTObjectController *controller;
   TVirtualPad *object;
}

@property (nonatomic, retain) UISwitch *logX;
@property (nonatomic, retain) UISwitch *logY;
@property (nonatomic, retain) UISwitch *logZ;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) logActivated : (UISwitch *) log;

@end
