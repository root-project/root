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

const NSWindowStyleMask kMiniaturizableWindowMask = NSWindowStyleMaskMiniaturizable;
const NSWindowStyleMask kResizableWindowMask = NSWindowStyleMaskResizable;
const NSWindowStyleMask kClosableWindowMask = NSWindowStyleMaskClosable;
const NSWindowStyleMask kTitledWindowMask = NSWindowStyleMaskTitled;
const NSWindowStyleMask kBorderlessWindowMask = NSWindowStyleMaskBorderless;

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

const NSWindowStyleMask kMiniaturizableWindowMask = NSMiniaturizableWindowMask;
const NSWindowStyleMask kResizableWindowMask = NSResizableWindowMask;
const NSWindowStyleMask kClosableWindowMask = NSClosableWindowMask;
const NSWindowStyleMask kTitledWindowMask = NSTitledWindowMask;
const NSWindowStyleMask kBorderlessWindowMask = NSBorderlessWindowMask;

#endif // MAC_OS_X_VERSION_10_12

}
}
}
