#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

class TObject;
class TAxis;

@interface AxisTitleInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UITextField *titleField;
   IBOutlet UISwitch *centered;
   IBOutlet UISwitch *rotated;

   ROOTObjectController *controller;
   TAxis *object;

   IBOutlet UILabel *offsetLabel;
   IBOutlet UIButton *plusOffsetBtn;
   IBOutlet UIButton *minusOffsetBtn;
   float offset;
   
   IBOutlet UILabel *sizeLabel;
   IBOutlet UIButton *plusSizeBtn;
   IBOutlet UIButton *minusSizeBtn;
   float titleSize;
}

@property (nonatomic, retain) UITextField *titleField;
@property (nonatomic, retain) UISwitch *centered;
@property (nonatomic, retain) UISwitch *rotated;
@property (nonatomic, retain) UILabel *offsetLabel;
@property (nonatomic, retain) UIButton *plusOffsetBtn;
@property (nonatomic, retain) UIButton *minusOffsetBtn;

@property (nonatomic, retain) UILabel *sizeLabel;
@property (nonatomic, retain) UIButton *plusSizeBtn;
@property (nonatomic, retain) UIButton *minusSizeBtn;

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) showTitleFontInspector;
- (IBAction) showTitleColorInspector;

- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;
- (IBAction) centerTitle;
- (IBAction) rotateTitle;
- (IBAction) plusOffset;
- (IBAction) minusOffset;
- (IBAction) plusSize;
- (IBAction) minusSize;

- (IBAction) back;

@end
