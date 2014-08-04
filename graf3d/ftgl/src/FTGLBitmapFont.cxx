#include "FTGLBitmapFont.h"
#include "FTBitmapGlyph.h"


FTGLBitmapFont::FTGLBitmapFont( const char* fontFilePath)
:   FTFont( fontFilePath)
{}


FTGLBitmapFont::FTGLBitmapFont( const unsigned char *pBufferBytes, size_t bufferSizeInBytes)
:   FTFont( pBufferBytes, bufferSizeInBytes)
{}


FTGLBitmapFont::~FTGLBitmapFont()
{}


FTGlyph* FTGLBitmapFont::MakeGlyph( unsigned int g)
{
    FT_GlyphSlot ftGlyph = face.Glyph( g, FT_LOAD_DEFAULT);

    if( ftGlyph)
    {
        FTBitmapGlyph* tempGlyph = new FTBitmapGlyph( ftGlyph);
        return tempGlyph;
    }

    err = face.Error();
    return NULL;
}


void FTGLBitmapFont::PreRender()
{
    FTFont::PreRender();
    glPushClientAttrib( GL_CLIENT_PIXEL_STORE_BIT);
    glPushAttrib( GL_ENABLE_BIT);

    glPixelStorei( GL_UNPACK_LSB_FIRST, GL_FALSE);
    glPixelStorei( GL_UNPACK_ALIGNMENT, 1);

    glDisable( GL_BLEND);
}


void FTGLBitmapFont::PostRender()
{
    glPopAttrib();
    glPopClientAttrib();
    FTFont::PostRender();
}
