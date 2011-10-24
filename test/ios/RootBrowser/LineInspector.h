#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"
#import "HorizontalPickerDelegate.h"


@class ROOTObjectController;
@class HorizontalPickerView;
@class LineWidthPicker;

class TAttLine;
class TObject;

//Line inspector is a composition of two sub-inspectors: line color and width inspector + 
//line style inspector.

@interface LineInspector : UIViewController <ObjectInspectorComponent, HorizontalPickerDelegate> {
@private
   NSMutableArray *lineStyles;
   NSMutableArray *lineColors;

   HorizontalPickerView *lineColorPicker;
   HorizontalPickerView *lineStylePicker;

   IBOutlet LineWidthPicker *lineWidthPicker;
   int lineWidth;

   ROOTObjectController *controller;
   TAttLine *object;
}

@property (nonatomic, retain) LineWidthPicker *lineWidthPicker;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;

- (void) item : (unsigned int)item wasSelectedInPicker : (HorizontalPickerView *)picker;

- (IBAction) decLineWidth;
- (IBAction) incLineWidth;

@end
