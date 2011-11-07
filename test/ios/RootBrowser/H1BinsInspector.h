#import "ObjectInspectorComponent.h"

@interface H1BinsInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UITextField *titleField;
   __weak IBOutlet UILabel *minLabel;
   __weak IBOutlet UILabel *maxLabel;
   __weak IBOutlet UISwitch *showMarkers;
}

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;
- (IBAction) toggleMarkers;


@end
