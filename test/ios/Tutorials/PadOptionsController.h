#import <CoreGraphics/CGPattern.h>
#import <UIKit/UIKit.h>

@class PadView;

namespace ROOT {
namespace iOS {

class Pad;

}
}


@interface PadOptionsController : UIViewController {
   __weak IBOutlet UISwitch *tickX_;
   __weak IBOutlet UISwitch *tickY_;

   __weak IBOutlet UISwitch *gridX_;
   __weak IBOutlet UISwitch *gridY_;

   __weak IBOutlet UISwitch *logX_;
   __weak IBOutlet UISwitch *logY_;
   __weak IBOutlet UISwitch *logZ_;
   
   __weak IBOutlet UIPickerView *colorPicker_;
   __weak IBOutlet UIPickerView *patternPicker_;
}

- (void) setView : (PadView *) view andPad : (ROOT::iOS::Pad *) pad;


- (IBAction) tickActivated : (id) control;
- (IBAction) gridActivated : (id) control;
- (IBAction) logActivated : (id) control;

@end
