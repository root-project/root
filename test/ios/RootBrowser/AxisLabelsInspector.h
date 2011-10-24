#import "ObjectInspectorComponent.h"

@class ROOTObjectController;


class TObject;
class TAxis;

@interface AxisLabelsInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UIButton *plusSize;
   IBOutlet UIButton *minusSize;
   IBOutlet UILabel *sizeLabel;
   
   IBOutlet UIButton *plusOffset;
   IBOutlet UIButton *minusOffset;
   IBOutlet UILabel *offsetLabel;
   
   IBOutlet UISwitch *noExp;
   
   ROOTObjectController *controller;
   TAxis *object;
}

@property (nonatomic, retain) UIButton *plusSize;
@property (nonatomic, retain) UIButton *minusSize;
@property (nonatomic, retain) UILabel *sizeLabel;
@property (nonatomic, retain) UIButton *plusOffset;
@property (nonatomic, retain) UIButton *minusOffset;
@property (nonatomic, retain) UILabel *offsetLabel;
@property (nonatomic, retain) UISwitch *noExp;

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) showLabelFontInspector;

- (IBAction) plusBtn : (UIButton *)sender;
- (IBAction) minusBtn : (UIButton *)sender;
- (IBAction) noExpPressed;
- (IBAction) back;

@end
