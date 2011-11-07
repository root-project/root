#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface PadLogScaleInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UISwitch *logX;
   __weak IBOutlet UISwitch *logY;
   __weak IBOutlet UISwitch *logZ;
}

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) logActivated : (UISwitch *) log;

@end
