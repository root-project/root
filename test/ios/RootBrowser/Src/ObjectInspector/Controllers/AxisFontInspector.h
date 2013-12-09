#import "ObjectInspectorComponent.h"

namespace ROOT_IOSObjectInspector {

enum AxisFontInspectorMode{
afimTitleFont,
afimLabelFont
};

}

@interface AxisFontInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent>

- (id)initWithNibName : (NSString *) nibName mode : (ROOT_IOSObjectInspector::AxisFontInspectorMode) m;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

- (IBAction) back;

@end
