#ifndef ROOT_TGLSdfFontMaker
#define ROOT_TGLSdfFontMaker

class TGLSdfFontMaker {
public:
   static int MakeFont(const char *ttf_font, const char *output_prefix, bool verbose = false);
};

#endif
