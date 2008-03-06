#ifndef BMP_H_HEADER_INCLUDED
#define BMP_H_HEADER_INCLUDED

#include "asimage.h"
#ifdef __cplusplus
extern "C" {
#endif

#define BMP_SIGNATURE		0x4D42             /* "BM" */

#if defined(_WIN32) && !defined(_WINGDI_)
#include <windows.h>
#endif

#ifndef _WINGDI_

typedef struct tagRGBQUAD { /* rgbq */ 
    CARD8    rgbBlue; 
    CARD8    rgbGreen; 
    CARD8    rgbRed; 
    CARD8    rgbReserved; 
} RGBQUAD; 

typedef struct tagBITMAPFILEHEADER {
	CARD16  bfType;
    CARD32  bfSize;
    CARD16  bfReserved1;
    CARD16  bfReserved2;
    CARD32  bfOffBits;
} BITMAPFILEHEADER;

typedef struct tagBITMAPINFOHEADER
{
	CARD32 biSize;
	CARD32 biWidth,  biHeight;
	CARD16 biPlanes, biBitCount;
	CARD32 biCompression;
	CARD32 biSizeImage;
	CARD32 biXPelsPerMeter, biYPelsPerMeter;
	CARD32 biClrUsed, biClrImportant;
}BITMAPINFOHEADER;

typedef struct tagBITMAPINFO { /* bmi */ 
    BITMAPINFOHEADER bmiHeader; 
    RGBQUAD          bmiColors[1]; 
} BITMAPINFO; 

#endif

#ifndef BI_RGB     
#define BI_RGB        0L
#endif


void 
dib_data_to_scanline( ASScanline *buf, 
                      BITMAPINFOHEADER *bmp_info, CARD8 *gamma_table, 
					  CARD8 *data, CARD8 *cmap, int cmap_entry_size); 

BITMAPINFO *
ASImage2DIB( ASVisual *asv, ASImage *im, 
		      int offset_x, int offset_y,
			   unsigned int to_width,
			   unsigned int to_height,
  			   void **pBits, int mask );
/* fixing a typo : */
#define ASImage2DBI ASImage2DIB

/* DIB colormap and data should follow the header as a continuous 
 * memory block !*/
ASImage *DIB2ASImage(BITMAPINFO *bmp_info, int compression);

ASImage      *
bitmap2asimage (unsigned char *xim, int width, int height,
                unsigned int compression, unsigned char *mask);

#ifdef __cplusplus
}
#endif

#endif

