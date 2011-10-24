#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

class TAttAxis;

namespace ROOT_IOSObjectInspector {

enum AxisColorInspectorMode {
   acimAxisColor,
   acimTitleColor,
   acimLabelColor
};

}
@interface AxisColorInspector : UIViewController <UIPickerViewDelegate, UIPickerViewDataSource, ObjectInspectorComponent> {
@private
   ROOT_IOSObjectInspector::AxisColorInspectorMode mode;

   IBOutlet UIPickerView *colorPicker;
   IBOutlet UILabel *titleLabel;
   
   NSMutableArray *colors;
   
   ROOTObjectController *controller;
   TAttAxis *object;
}

@property (nonatomic, retain) UIPickerView *colorPicker;
@property (nonatomic, retain) UILabel *titleLabel;

- (id) initWithNibName : (NSString *)nibNameOrNil bundle : (NSBundle *)nibBundleOrNil mode : (ROOT_IOSObjectInspector::AxisColorInspectorMode)mode;

- (void) setROOTObjectController : (ROOTObjectController *)contoller;
- (void) setROOTObject : (TObject *)object;

- (IBAction) back;

@end
