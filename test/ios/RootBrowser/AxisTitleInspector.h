#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface AxisTitleInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UITextField *titleField;
   __weak IBOutlet UISwitch *centered;
   __weak IBOutlet UISwitch *rotated;
   __weak IBOutlet UILabel *offsetLabel;
   __weak IBOutlet UILabel *sizeLabel;
   __weak IBOutlet UIButton *plusSizeBtn;
   __weak IBOutlet UIButton *minusSizeBtn;
}

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) showTitleFontInspector;
- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;
- (IBAction) centerTitle;
- (IBAction) rotateTitle;
- (IBAction) plusOffset;
- (IBAction) minusOffset;
- (IBAction) plusSize;
- (IBAction) minusSize;

@end
