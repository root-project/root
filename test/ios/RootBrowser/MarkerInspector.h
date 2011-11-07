#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@interface MarkerInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate> {
@private
   __weak IBOutlet UIButton *plusBtn;
   __weak IBOutlet UIButton *minusBtn;
   __weak IBOutlet UILabel *sizeLabel;
}

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;

- (IBAction) plusPressed;
- (IBAction) minusPressed;

@end
