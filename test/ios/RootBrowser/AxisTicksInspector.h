#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;

//TAxis, not TAttAxis, since inspector has to work with
//several functions, which are member of TAxis, not TAttAxis.
class TAxis;

@interface AxisTicksInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UILabel *tickLengthLabel;
   IBOutlet UIButton *plusLengthBtn;
   IBOutlet UIButton *minusLengthBtn;
   float tickLength;

   IBOutlet UIButton *plusPrim;
   IBOutlet UIButton *minusPrim;
   IBOutlet UILabel *primLabel;
   unsigned primaryTicks;

   IBOutlet UIButton *plusSec;
   IBOutlet UIButton *minusSec;
   IBOutlet UILabel *secLabel;
   unsigned secondaryTicks;

   IBOutlet UIButton *plusTer;
   IBOutlet UIButton *minusTer;
   IBOutlet UILabel *terLabel;
   unsigned tertiaryTicks;

   IBOutlet UISegmentedControl *ticksNegPos;
   
   ROOTObjectController *controller;
   TAxis *object;
}

@property (nonatomic, retain) UILabel *tickLengthLabel;
@property (nonatomic, retain) UIButton *plusLengthBtn;
@property (nonatomic, retain) UIButton *minusLengthBtn;

@property (nonatomic, retain) UIButton *plusPrim;
@property (nonatomic, retain) UIButton *minusPrim;
@property (nonatomic, retain) UILabel *primLabel;

@property (nonatomic, retain) UIButton *plusSec;
@property (nonatomic, retain) UIButton *minusSec;
@property (nonatomic, retain) UILabel *secLabel;

@property (nonatomic, retain) UIButton *plusTer;
@property (nonatomic, retain) UIButton *minusTer;
@property (nonatomic, retain) UILabel *terLabel;

@property (nonatomic, retain) UISegmentedControl *ticksNegPos;

- (void) setROOTObject : (TObject *)object;
- (void) setROOTObjectController : (ROOTObjectController *)c;

- (IBAction) plusTick : (UIButton *)sender;
- (IBAction) minusTick :(UIButton *)sender;
- (IBAction) ticksNegPosPressed;

- (IBAction) back;

@end
