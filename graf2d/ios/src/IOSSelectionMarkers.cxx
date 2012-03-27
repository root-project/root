#include "IOSResourceManagement.h"
#include "IOSSelectionMarkers.h"

namespace ROOT {
namespace iOS {
namespace GraphicUtils {

//______________________________________________________________________________
void DrawSelectionMarker(CGContextRef ctx, const CGPoint &point)
{
   //
   const CGFloat externalRadius = 5.f;
   const CGFloat internalRadius = 4.f;

   const Util::CGStateGuard ctxGuard(ctx);
   CGContextSetRGBFillColor(ctx, 1.f, 1.f, 1.f, 1.f);
   CGContextFillEllipseInRect(ctx, CGRectMake(point.x - externalRadius, point.y - externalRadius, 2 * externalRadius, 2 * externalRadius));
   CGContextSetRGBFillColor(ctx, 0.f, 0.f, 1.f, 1.f);
   CGContextFillEllipseInRect(ctx, CGRectMake(point.x - internalRadius, point.y - internalRadius, 2 * internalRadius, 2 * internalRadius));
}

//______________________________________________________________________________
void DrawBoxSelectionMarkers(CGContextRef ctx, const CGRect &box)
{
   //
   DrawSelectionMarker(ctx, box.origin);
   DrawSelectionMarker(ctx, CGPointMake(box.origin.x, box.origin.y + box.size.height));
   DrawSelectionMarker(ctx, CGPointMake(box.origin.x + box.size.width, box.origin.y));
   DrawSelectionMarker(ctx, CGPointMake(box.origin.x + box.size.width, box.origin.y + box.size.height));
}

}
}
}