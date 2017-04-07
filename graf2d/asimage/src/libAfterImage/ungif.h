#ifndef UNGIF_H_HEADER_INCLUDED
#define UNGIF_H_HEADER_INCLUDED

#ifdef HAVE_GIF		/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */

#ifdef __cplusplus
extern "C" {
#endif

#if ((GIFLIB_MAJOR==4) && (GIFLIB_MINOR>=2)) 
static inline void PrintGifError(void) {
    fprintf(stderr, "%s\n", GifErrorString());
}
#elif (GIFLIB_MAJOR>=5)
static inline void PrintGifError(int code) {
    fprintf(stderr, "%s\n", GifErrorString(code));
}
#endif

#if (GIFLIB_MAJOR>=5)
#ifdef __GNUC__
#define ASIM_PrintGifError(code) do{ fprintf( stderr, "%s():%d:<%s> ",__FUNCTION__, __LINE__, path?path:"null" ); PrintGifError(code); }while(0)
#else
#define ASIM_PrintGifError(code) do{ PrintGifError(code); }while(0)
#endif
#else // (GIFLIB_MAJOR>=5)
#ifdef __GNUC__
#define ASIM_PrintGifError() do{ fprintf( stderr, "%s():%d:<%s> ",__FUNCTION__, __LINE__, path?path:"null" ); PrintGifError(); }while(0)
#else
#define ASIM_PrintGifError() do{ PrintGifError(); }while(0)
#endif
#endif // (GIFLIB_MAJOR>=5)

#define GIF_GCE_DELAY_BYTE_LOW	1
#define GIF_GCE_DELAY_BYTE_HIGH	2
#define GIF_GCE_TRANSPARENCY_BYTE	3
#define GIF_NETSCAPE_REPEAT_BYTE_LOW	1
#define GIF_NETSCAPE_REPEAT_BYTE_HIGH	2

void free_gif_saved_image( SavedImage *sp, Bool reusable );
void free_gif_saved_images( SavedImage *images, int count );


int fread_gif( GifFileType *gif, GifByteType* buf, int len );
#if (GIFLIB_MAJOR>=5)
GifFileType* open_gif_read( FILE *in_stream, int *errcode );
#else
GifFileType* open_gif_read( FILE *in_stream );
#endif

int get_gif_image_desc( GifFileType *gif, SavedImage *im );

int get_gif_saved_images( GifFileType *gif, int subimage, SavedImage **ret, int *ret_images  );

int write_gif_saved_images( GifFileType *gif, SavedImage *images, unsigned int count );

#ifdef __cplusplus
}
#endif

#endif			/* GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF GIF */


#endif
