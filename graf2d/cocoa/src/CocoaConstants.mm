#include "CocoaConstants.h"

namespace ROOT {
namespace MacOSX {
namespace Details {

#ifdef MAC_OS_X_VERSION_10_12

const NSUInteger kEventMaskAny = NSEventMaskAny;
const NSUInteger kAlphaShiftKeyMask = NSEventModifierFlagCapsLock;
const NSUInteger kShiftKeyMask = NSEventModifierFlagShift;
const NSUInteger kControlKeyMask = NSEventModifierFlagControl;
const NSUInteger kAlternateKeyMask = NSEventModifierFlagOption;
const NSUInteger kCommandKeyMask = NSEventModifierFlagCommand;
const NSUInteger kDeviceIndependentModifierFlagsMask = NSEventModifierFlagDeviceIndependentFlagsMask;

const NSEventType kKeyDown = NSEventTypeKeyDown;
const NSEventType kKeyUp = NSEventTypeKeyUp;

const NSEventType kLeftMouseDown = NSEventTypeLeftMouseDown;
const NSEventType kRightMouseDown = NSEventTypeRightMouseDown;

const NSEventType kApplicationDefined = NSEventTypeApplicationDefined;

const NSUInteger kMiniaturizableWindowMask = NSWindowStyleMaskMiniaturizable;
const NSUInteger kResizableWindowMask = NSWindowStyleMaskResizable;
const NSUInteger kClosableWindowMask = NSWindowStyleMaskClosable;
const NSUInteger kTitledWindowMask = NSWindowStyleMaskTitled;
const NSUInteger kBorderlessWindowMask = NSWindowStyleMaskBorderless;

#else

const NSUInteger kEventMaskAny = NSAnyEventMask;
const NSUInteger kAlphaShiftKeyMask = NSAlphaShiftKeyMask;
const NSUInteger kShiftKeyMask = NSShiftKeyMask;
const NSUInteger kControlKeyMask = NSControlKeyMask;
const NSUInteger kAlternateKeyMask = NSAlternateKeyMask;
const NSUInteger kCommandKeyMask = NSCommandKeyMask;
const NSUInteger kDeviceIndependentModifierFlagsMask = NSDeviceIndependentModifierFlagsMask;

const NSEventType kKeyDown = NSKeyDown;
const NSEventType kKeyUp = NSKeyUp;

const NSEventType kLeftMouseDown = NSLeftMouseDown;
const NSEventType kRightMouseDown = NSRightMouseDown;

const NSEventType kApplicationDefined = NSApplicationDefined;

const NSUInteger kMiniaturizableWindowMask = NSMiniaturizableWindowMask;
const NSUInteger kResizableWindowMask = NSResizableWindowMask;
const NSUInteger kClosableWindowMask = NSClosableWindowMask;
const NSUInteger kTitledWindowMask = NSTitledWindowMask;
const NSUInteger kBorderlessWindowMask = NSBorderlessWindowMask;

#endif // MAC_OS_X_VERSION_10_12

}
}
}
