#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@class HorizontalPickerView;

class TAttMarker;

@interface MarkerInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate> {
@private   
   HorizontalPickerView *markerStylePicker;
   HorizontalPickerView *markerColorPicker;

   NSMutableArray *styleCells;
   NSMutableArray *colorCells;
   
   IBOutlet UIButton *plusBtn;
   IBOutlet UIButton *minusBtn;
   IBOutlet UILabel *sizeLabel;
   
   ROOTObjectController *controller;
   TAttMarker *object;
}

@property (nonatomic, retain) UIButton *plusBtn;
@property (nonatomic, retain) UIButton *minusBtn;
@property (nonatomic, retain) UILabel *sizeLabel;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;

- (IBAction) plusPressed;
- (IBAction) minusPressed;

@end
