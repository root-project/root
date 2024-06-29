#ifndef ROOT_TGLSdfFontMaker
#define ROOT_TGLSdfFontMaker

#include "Rtypes.h"

class TGLSdfFontMaker {
public:
    static int MakeFont(const char* ttf_font, const char* output_prefix, bool verbose=false);

    ClassDef(TGLSdfFontMaker, 0); 
};

#endif
