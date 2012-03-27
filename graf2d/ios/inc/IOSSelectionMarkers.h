#ifndef ROOT_IOSSelectionMarkers
#define ROOT_IOSSelectionMarkers

#include <CoreGraphics/CGGeometry.h>
#include <CoreGraphics/CGContext.h>

namespace ROOT {
namespace iOS {
namespace GraphicUtils {

void DrawSelectionMarker(CGContextRef ctx, const CGPoint &point);
void DrawBoxSelectionMarkers(CGContextRef ctx, const CGRect &box);

}
}
}

#endif
