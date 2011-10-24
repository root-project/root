#import "ObjectInspectorComponent.h"

class TH1;

@interface H1ErrorsInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent> {
@private
   IBOutlet UIPickerView *errorTypePicker;

   ROOTObjectController *controller;
   TH1 *object;
}

@property (nonatomic, retain) UIPickerView *errorTypePicker;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

@end
