#ifndef ROOT_CocoaConstants
#define ROOT_CocoaConstants

#include <Cocoa/Cocoa.h>

/*
While releasing a new SDK version Apple sometimes deprecates some constants,
creating a lot of compilation noise as a result. We can not just use the recommended
new values - we support the previous SDK versions where these constants do not exist yet.
We define our own constants that will resolve at compile time to either new or
old ones. All similar constants should be added here.
Resulting kNames are quite generic and can clash with ROOT's own constants
(for example, kShiftKeyMask is very similar to ROOTS kKeyShiftMask),
but they are protected by the namepace scopes.
*/

namespace ROOT {
namespace MacOSX {
namespace Details {

// Key modifiers
extern const NSUInteger kEventMaskAny;
extern const NSUInteger kAlphaShiftKeyMask;
extern const NSUInteger kShiftKeyMask;
extern const NSUInteger kControlKeyMask;
extern const NSUInteger kAlternateKeyMask;
extern const NSUInteger kCommandKeyMask;
extern const NSUInteger kDeviceIndependentModifierFlagsMask;

// Application event types
extern const NSEventType kApplicationDefined;

// Key event types
extern const NSEventType kKeyDown;
extern const NSEventType kKeyUp;

// Mouse events
extern const NSEventType kLeftMouseDown;
extern const NSEventType kRightMouseDown;

// Windows masks
extern const NSUInteger kMiniaturizableWindowMask;
extern const NSUInteger kResizableWindowMask;
extern const NSUInteger kClosableWindowMask;
extern const NSUInteger kTitledWindowMask;
extern const NSUInteger kBorderlessWindowMask;


}
}
}

#endif
