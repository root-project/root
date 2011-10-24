#import "ObjectInspectorComponent.h"

@class H1ErrorsInspector;
@class H1BinsInspector;

@interface H1Inspector : UIViewController <ObjectInspectorComponent> {
@private
   IBOutlet UITabBar *tabBar;
   
   H1ErrorsInspector *errorInspector;
   H1BinsInspector *binsInspector;
   
   TObject *object;
   ROOTObjectController *controller;
}

@property (nonatomic, retain) UITabBar *tabBar;

- (void) setROOTObject : (TObject *)o;
- (void) setROOTObjectController : (ROOTObjectController *)c;
- (NSString *) getComponentName;
- (void) resetInspector;

- (void) showBinsInspector;
- (void) showErrorInspector;

@end
