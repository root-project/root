#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface PadTicksGridInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UISwitch *gridX;
   __weak IBOutlet UISwitch *gridY;
   __weak IBOutlet UISwitch *ticksX;
   __weak IBOutlet UISwitch *ticksY;
}

- (void) setROOTObjectController : (ROOTObjectController *) c;
- (void) setROOTObject : (TObject *) obj;

- (IBAction) gridActivated : (UISwitch *) g;
- (IBAction) ticksActivated : (UISwitch *) t;

@end
