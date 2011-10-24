#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

class TVirtualPad;
class TObject;

@interface PadTicksGridInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UISwitch *gridX;
   IBOutlet UISwitch *gridY;
   IBOutlet UISwitch *ticksX;
   IBOutlet UISwitch *ticksY;
   
   ROOTObjectController *controller;
   TVirtualPad *object;
}

@property (nonatomic, retain) UISwitch *gridX;
@property (nonatomic, retain) UISwitch *gridY;
@property (nonatomic, retain) UISwitch *ticksX;
@property (nonatomic, retain) UISwitch *ticksY;

- (void) setROOTObjectController : (ROOTObjectController *) c;
- (void) setROOTObject : (TObject *) obj;

- (IBAction) gridActivated : (UISwitch *) g;
- (IBAction) ticksActivated : (UISwitch *) t;

@end
