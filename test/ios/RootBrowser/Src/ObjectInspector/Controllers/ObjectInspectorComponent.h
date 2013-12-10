#import <Foundation/Foundation.h>

//
//Protocol to be adopted by every specific "object editor" or
//"object-inspector".
//

@class ObjectViewController;

class TObject;

@protocol ObjectInspectorComponent <NSObject>
@required

- (void) setObjectController : (ObjectViewController *) c;
- (void) setObject : (TObject *) o;

@optional

- (NSString*) getComponentName;
- (void) resetInspector;

@end
