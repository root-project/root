#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

class TObject;
class TAxis;

namespace ROOT_IOSObjectInspector {

enum AxisFontInspectorMode{
afimTitleFont,
afimLabelFont
};

}

@interface AxisFontInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent> {
@private
   IBOutlet UILabel *titleLabel;
   IBOutlet UIPickerView *fontPicker;
   
   ROOT_IOSObjectInspector::AxisFontInspectorMode mode;
   
   ROOTObjectController *controller;
   TAxis *object;
}

@property (nonatomic, retain) UILabel *titleLabel;
@property (nonatomic, retain) UIPickerView *fontPicker;


- (id)initWithNibName : (NSString *)nibName mode : (ROOT_IOSObjectInspector::AxisFontInspectorMode)m;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) back;

@end
