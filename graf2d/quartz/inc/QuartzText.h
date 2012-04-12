//Author: Timur Pocheptsov.
#ifndef ROOT_QuartzText
#define ROOT_QuartzText

#include <vector>

//This must be changed: different header for iOS and MacOSX.
//#include <ApplicationServices/ApplicationServices.h>
#include <Cocoa/Cocoa.h>

#include "CocoaUtils.h"
#include "GuiTypes.h"


namespace ROOT {
namespace Quartz {
   
// Core Text's CTLine wrapper.
class TextLine {
public:
   TextLine(const char *textLine, CTFontRef font);
   TextLine(const std::vector<UniChar> &textLine, CTFontRef font);
   
   TextLine(const char *textLine, CTFontRef font, Color_t color);
   TextLine(const std::vector<UniChar> &textLine, CTFontRef font, Color_t color);
   TextLine(const char *textLine, CTFontRef font, const CGFloat *rgb);

   ~TextLine();

   void GetBounds(UInt_t &w, UInt_t &h)const;
   void GetAscentDescent(Int_t &asc, Int_t &desc)const;

   void DrawLine(CGContextRef ctx)const;
   void DrawLine(CGContextRef ctx, Double_t x, Double_t y)const;
private:
   CTLineRef fCTLine; //Core Text line, created from Attributed string.

   void Init(const char *textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);
   void Init(const std::vector<UniChar> &textLine, UInt_t nAttribs, CFStringRef *keys, CFTypeRef *values);

   TextLine(const TextLine &rhs) = delete;
   TextLine &operator = (const TextLine &rhs) = delete;
};

//Aux. function which extracts glyphs and calculates their positions.
void DrawTextLineNoKerning(CGContextRef ctx, CTFontRef font, const std::vector<UniChar> &text, Int_t x, Int_t y);

}
}

#endif
