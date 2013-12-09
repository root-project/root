#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

@interface PadInspector : UIViewController <ObjectInspectorComponent, UITabBarDelegate>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *)c;
- (void) setObject : (TObject *)o;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
