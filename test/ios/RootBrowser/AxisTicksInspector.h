#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface AxisTicksInspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UILabel *tickLengthLabel;
   __weak IBOutlet UIButton *plusLengthBtn;
   __weak IBOutlet UIButton *minusLengthBtn;

   __weak IBOutlet UIButton *plusPrim;
   __weak IBOutlet UIButton *minusPrim;
   __weak IBOutlet UILabel *primLabel;

   __weak IBOutlet UIButton *plusSec;
   __weak IBOutlet UIButton *minusSec;
   __weak IBOutlet UILabel *secLabel;

   __weak IBOutlet UIButton *plusTer;
   __weak IBOutlet UIButton *minusTer;
   __weak IBOutlet UILabel *terLabel;

   __weak IBOutlet UISegmentedControl *ticksNegPos;
}

- (void) setROOTObject : (TObject *)object;
- (void) setROOTObjectController : (ROOTObjectController *)c;

- (IBAction) plusTick : (UIButton *)sender;
- (IBAction) minusTick :(UIButton *)sender;
- (IBAction) ticksNegPosPressed;

@end
