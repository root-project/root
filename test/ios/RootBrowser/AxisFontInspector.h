#import "ObjectInspectorComponent.h"

namespace ROOT_IOSObjectInspector {

enum AxisFontInspectorMode{
afimTitleFont,
afimLabelFont
};

}

@interface AxisFontInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent> {
@private
   __weak IBOutlet UILabel *titleLabel;
   __weak IBOutlet UIPickerView *fontPicker;
}

- (id)initWithNibName : (NSString *)nibName mode : (ROOT_IOSObjectInspector::AxisFontInspectorMode)m;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) back;

@end
