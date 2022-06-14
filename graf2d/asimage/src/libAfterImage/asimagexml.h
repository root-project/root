#ifndef ASIMAGEXML_HEADER_FILE_INCLUDED
#define ASIMAGEXML_HEADER_FILE_INCLUDED

#include "asimage.h"

#ifdef __cplusplus
extern "C" {
#endif

/* We don't trust the math library to actually provide this number.*/
#undef PI
#define PI 180


#define ASIM_XML_ENABLE_SAVE 	(0x01<<0)
#define ASIM_XML_ENABLE_SHOW 	(0x01<<1)


struct ASImageManager ;
struct ASFontManager ;
struct xml_elem_t;

void set_xml_image_manager( struct ASImageManager *imman );
void set_xml_font_manager( struct ASFontManager *fontman );
struct ASImageManager *create_generic_imageman(const char *path);		   
struct ASFontManager *create_generic_fontman(Display *dpy, const char *path);


ASImage *
compose_asimage_xml(ASVisual *asv,
                    struct ASImageManager *imman,
					struct ASFontManager *fontman,
					char *doc_str, ASFlagType flags,
					int verbose, Window display_win,
					const char *path);


#define ASXMLVAR_TargetWidth 		"target.width"
#define ASXMLVAR_TargetHeight 		"target.height"


ASImage *
compose_asimage_xml_at_size(ASVisual *asv, 
							struct ASImageManager *imman, 
							struct ASFontManager *fontman, 
							char *doc_str, 
							ASFlagType flags, 
							int verbose, 
							Window display_win, 
							const char *path, 
							int target_width, 
							int target_height);

ASImage *
compose_asimage_xml_from_doc(ASVisual *asv, 
							 struct ASImageManager *imman, 
							 struct ASFontManager *fontman, 
							 struct xml_elem_t* doc, 
							 ASFlagType flags, 
							 int verbose, 
							 Window display_win, 
							 const char *path, 
							 int target_width, int target_height);

void show_asimage(ASVisual *asv, ASImage* im, Window w, long delay);
ASImage* build_image_from_xml( ASVisual *asv,
                               struct ASImageManager *imman,
							   struct ASFontManager *fontman,
							   struct xml_elem_t* doc, struct xml_elem_t** rparm,
							   ASFlagType flags, int verbose, Window display_win);
Bool save_asimage_to_file(const char* file2bsaved, ASImage *im,
	    			      const char* strtype,
						  const char *compress,
						  const char *opacity,
			  			  int delay, int replace);


#ifdef __cplusplus
}
#endif

#endif /*#ifndef ASIMAGEXML_HEADER_FILE_INCLUDED*/


