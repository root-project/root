#import "ObjectInspectorComponent.h"

@class RangeSlider;

class TH1;

@interface H1BinsInspector : UIViewController <ObjectInspectorComponent> {
@private
   RangeSlider *axisRangeSlider;
      
   IBOutlet UITextField *titleField;

   IBOutlet UILabel *minLabel;
   IBOutlet UILabel *maxLabel;
   
   IBOutlet UISwitch *showMarkers;
   
   ROOTObjectController *controller;
   TH1 *object;
}

@property (nonatomic, retain) UISwitch *showMarkers;
@property (nonatomic, retain) UITextField *titleField;
@property (nonatomic, retain) UILabel *minLabel;
@property (nonatomic, retain) UILabel *maxLabel;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

- (IBAction) textFieldDidEndOnExit : (id) sender;
- (IBAction) textFieldEditingDidEnd : (id) sender;
- (IBAction) toggleMarkers;


@end
