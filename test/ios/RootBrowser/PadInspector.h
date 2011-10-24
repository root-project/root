#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class ROOTObjectController;
@class PadTicksGridInspector;
@class PadLogScaleInspector;

class TObject;

@interface PadInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UITabBar *tabBar;

   PadTicksGridInspector *gridInspector;
   PadLogScaleInspector *logScaleInspector;
   
   ROOTObjectController *controller;
   TObject *object;
}

@property (nonatomic, retain) UITabBar *tabBar;

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;
- (void) resetInspector;

- (IBAction) showTicksAndGridInspector;
- (IBAction) showLogScaleInspector;

@end
