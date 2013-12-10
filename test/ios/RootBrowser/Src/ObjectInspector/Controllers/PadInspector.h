#import <UIKit/UIKit.h>

#import "ObjectInspectorComponent.h"

//
//Object inspector for a TAttPad, contains two tabs for: 1) axis ticks and grids and
//2) log scales.
//

@interface PadInspector : UIViewController <ObjectInspectorComponent, UITabBarDelegate>

+ (CGRect) inspectorFrame;

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;
- (NSString *) getComponentName;
- (void) resetInspector;

@end
