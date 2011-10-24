#import <Foundation/Foundation.h>

@class ROOTObjectController;

class TObject;

@protocol ObjectInspectorComponent <NSObject>

- (void) setROOTObjectController : (ROOTObjectController *)c;
- (void) setROOTObject : (TObject *)o;

@optional

- (NSString*) getComponentName;
- (void) resetInspector;

@end
