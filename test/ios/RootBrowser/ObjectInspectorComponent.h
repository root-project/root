#import <Foundation/Foundation.h>

//
//Protocol to be adopted by every specific "object editor" or
//"object-inspector".
//

@class ObjectViewController;

class TObject;

@protocol ObjectInspectorComponent <NSObject>

- (void) setROOTObjectController : (ObjectViewController *)c;
- (void) setROOTObject : (TObject *)o;

@optional

- (NSString*) getComponentName;
- (void) resetInspector;

@end
