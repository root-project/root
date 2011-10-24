#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"

@class ROOTObjectController;
@class HorizontalPickerView;

class TAttFill;
class TObject;

@interface FilledAreaInspector : UIViewController <HorizontalPickerDelegate, ObjectInspectorComponent> {
@private
   HorizontalPickerView *colorPicker;
   HorizontalPickerView *patternPicker;
   
   NSMutableArray *colorCells;
   NSMutableArray *patternCells;
   
   TAttFill *filledObject;
   ROOTObjectController *parentController;
}


- (void) setROOTObjectController : (ROOTObjectController *) p;
- (void) setROOTObject : (TObject*) obj;
- (NSString *) getComponentName;


@end
