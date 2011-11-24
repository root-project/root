#import <Foundation/Foundation.h>

//
//Protocol to be adopted by every specific "object editor" or
//"object-inspector".
//

@class ROOTObjectController;

class TObject;

@protocol ObjectInspectorComponent <NSObject>

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

@optional

- (NSString*) getComponentName;
- (void) resetInspector;

@end
