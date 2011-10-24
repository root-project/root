#import <CoreGraphics/CGPattern.h>
#import <UIKit/UIKit.h>

@class PadView;

namespace ROOT {
namespace iOS {

class Pad;

}
}


@interface PadOptionsController : UIViewController {
   IBOutlet UISwitch *tickX_;
   IBOutlet UISwitch *tickY_;

   IBOutlet UISwitch *gridX_;
   IBOutlet UISwitch *gridY_;

   IBOutlet UISwitch *logX_;
   IBOutlet UISwitch *logY_;
   IBOutlet UISwitch *logZ_;
   
   IBOutlet UIPickerView *colorPicker_;
   IBOutlet UIPickerView *patternPicker_;
   
   NSMutableArray *colors_;
   NSMutableArray *patterns_;
   
   ROOT::iOS::Pad *pad;
   PadView *padView;
}

@property (nonatomic, retain) UISwitch *tickX;
@property (nonatomic, retain) UISwitch *tickY;
@property (nonatomic, retain) UISwitch *gridX;
@property (nonatomic, retain) UISwitch *gridY;
@property (nonatomic, retain) UISwitch *logX;
@property (nonatomic, retain) UISwitch *logY;
@property (nonatomic, retain) UISwitch *logZ;
@property (nonatomic, retain) UIPickerView *colorPicker;
@property (nonatomic, retain) UIPickerView *patternPicker;
@property (nonatomic, retain) NSMutableArray *colors;
@property (nonatomic, retain) NSMutableArray *patterns;

- (void) setView : (PadView *) view andPad : (ROOT::iOS::Pad *) pad;


- (IBAction) tickActivated : (id) control;
- (IBAction) gridActivated : (id) control;
- (IBAction) logActivated : (id) control;

@end
