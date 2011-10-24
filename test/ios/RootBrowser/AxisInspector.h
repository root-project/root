#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@class InspectorWithNavigation;
@class AxisTicksInspector;
@class ROOTObjectController;

class TObject;

@interface AxisInspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UITabBar *tabBar;

   AxisTicksInspector *ticksInspector;

   InspectorWithNavigation *titleInspector;
   InspectorWithNavigation *labelInspector;

   ROOTObjectController *controller;
   TObject *object;
}

@property (nonatomic, retain) UITabBar *tabBar;

+ (CGRect) inspectorFrame;

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;
- (NSString *) getComponentName;
- (void) resetInspector;

- (IBAction) showTicksInspector;
- (IBAction) showAxisTitleInspector;
- (IBAction) showAxisLabelsInspector;

@end
