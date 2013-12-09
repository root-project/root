#import "ObjectInspectorComponent.h"

@interface H1ErrorsInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent> {
@private
   __weak IBOutlet UIPickerView *errorTypePicker;
}

//@property (nonatomic, retain) UIPickerView *errorTypePicker;

- (void) setROOTObjectController : (ObjectViewController *)c;
- (void) setROOTObject : (TObject *)o;

@end
