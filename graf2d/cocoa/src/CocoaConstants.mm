#include "CocoaConstants.h"

namespace ROOT {
namespace MacOSX {
namespace Details {

#ifdef MAC_OS_X_VERSION_10_12

const NSEventModifierFlags kAlphaShiftKeyMask = NSEventModifierFlagCapsLock;
const NSEventModifierFlags kShiftKeyMask = NSEventModifierFlagShift;
const NSEventModifierFlags kControlKeyMask = NSEventModifierFlagControl;
const NSEventModifierFlags kAlternateKeyMask = NSEventModifierFlagOption;
const NSEventModifierFlags kCommandKeyMask = NSEventModifierFlagCommand;
const NSEventModifierFlags kDeviceIndependentModifierFlagsMask = NSEventModifierFlagDeviceIndependentFlagsMask;

const NSEventType kKeyDown = NSEventTypeKeyDown;
const NSEventType kKeyUp = NSEventTypeKeyUp;

const NSEventType kLeftMouseDown = NSEventTypeLeftMouseDown;
const NSEventType kRightMouseDown = NSEventTypeRightMouseDown;

const NSUInteger kMiniaturizableWindowMask = NSWindowStyleMaskMiniaturizable;
const NSUInteger kResizableWindowMask = NSWindowStyleMaskResizable;
const NSUInteger kClosableWindowMask = NSWindowStyleMaskClosable;
const NSUInteger kTitledWindowMask = NSWindowStyleMaskTitled;
const NSUInteger kBorderlessWindowMask = NSWindowStyleMaskBorderless;

#else

const NSEventModifierFlags kAlphaShiftKeyMask = NSAlphaShiftKeyMask;
const NSEventModifierFlags kShiftKeyMask = NSShiftKeyMask;
const NSEventModifierFlags kControlKeyMask = NSControlKeyMask;
const NSEventModifierFlags kAlternateKeyMask = NSAlternateKeyMask;
const NSEventModifierFlags kCommandKeyMask = NSCommandKeyMask;
const NSEventModifierFlags kDeviceIndependentModifierFlagsMask = NSDeviceIndependentModifierFlagsMask;

const NSEventType kKeyDown = NSKeyDown;
const NSEventType kKeyUp = NSKeyUp;

const NSEventType kLeftMouseDown = NSLeftMouseDown;
const NSEventType kRightMouseDown = NSRightMouseDown;

const NSUInteger kMiniaturizableWindowMask = NSMiniaturizableWindowMask;
const NSUInteger kResizableWindowMask = NSResizableWindowMask;
const NSUInteger kClosableWindowMask = NSClosableWindowMask;
const NSUInteger kTitledWindowMask = NSTitledWindowMask;
const NSUInteger kBorderlessWindowMask = NSBorderlessWindowMask;

#endif // MAC_OS_X_VERSION_10_12

}
}
}
