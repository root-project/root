#import "ObjectInspectorComponent.h"

@interface H1Inspector : UIViewController <ObjectInspectorComponent> {
@private
   __weak IBOutlet UITabBar *tabBar;
}

- (void) setROOTObject : (TObject *)o;
- (void) setROOTObjectController : (ObjectViewController *)c;
- (NSString *) getComponentName;
- (void) resetInspector;


@end
