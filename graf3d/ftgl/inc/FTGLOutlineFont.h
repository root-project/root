#ifndef     __FTGLOutlineFont__
#define     __FTGLOutlineFont__


#include "FTFont.h"
#include "FTGL.h"

class FTGlyph;


/**
 * FTGLOutlineFont is a specialisation of the FTFont class for handling
 * Vector Outline fonts
 *
 * @see     FTFont
 */
class FTGL_EXPORT FTGLOutlineFont : public FTFont
{
    public:
        /**
         * Open and read a font file. Sets Error flag.
         *
         * @param fontFilePath  font file path.
         */
        FTGLOutlineFont( const char* fontFilePath);

        /**
         * Open and read a font from a buffer in memory. Sets Error flag.
         *
         * @param pBufferBytes  the in-memory buffer
         * @param bufferSizeInBytes  the length of the buffer in bytes
         */
        FTGLOutlineFont( const unsigned char *pBufferBytes, size_t bufferSizeInBytes);

        /**
         * Destructor
         */
        ~FTGLOutlineFont();

        /**
         * Prepare for rendering
         */
        virtual void PreRender() override;

        /**
         * Cleanup after rendering
         */
        virtual void PostRender() override;

    private:
        /**
         * Construct a FTOutlineGlyph.
         *
         * @param g The glyph index NOT the char code.
         * @return  An FTOutlineGlyph or <code>null</code> on failure.
         */
        inline virtual FTGlyph* MakeGlyph( unsigned int g) override;

};
#endif // __FTGLOutlineFont__
