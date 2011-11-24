#import "ObjectInspectorComponent.h"

@class ROOTObjectController;


class TObject;
class TAxis;

@interface AxisLabelsInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UIButton *plusSize;
   __weak IBOutlet UIButton *minusSize;
   __weak IBOutlet UILabel *sizeLabel;
   
   __weak IBOutlet UIButton *plusOffset;
   __weak IBOutlet UIButton *minusOffset;
   __weak IBOutlet UILabel *offsetLabel;
   
   __weak IBOutlet UISwitch *noExp;
}

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) showLabelFontInspector;

- (IBAction) plusBtn : (UIButton *)sender;
- (IBAction) minusBtn : (UIButton *)sender;
- (IBAction) noExpPressed;

@end
