#import <Foundation/Foundation.h>

@class HorizontalPickerView;

@protocol HorizontalPickerDelegate <NSObject>

- (void) item : (unsigned)item wasSelectedInPicker : (HorizontalPickerView *)picker;

@end
