#ifndef LIB_AFTERIMAGE_H_HEADER_INCLUDED
#define LIB_AFTERIMAGE_H_HEADER_INCLUDED

/* the follwoing files has to be included in user app to get access to
 * libAfterImage functionality.
 */
#include "afterbase.h"
#include "asvisual.h"
#include "blender.h"
#include "asimage.h"
#include "imencdec.h"
#include "ascmap.h"
#undef HAVE_FREETYPE
#include "asfont.h"
#include "ximage.h"
#include "transform.h"
#include "asimagexml.h"
#include "import.h"
#include "export.h"
#include "pixmap.h"
#include "char2uni.h"

/****h* libAfterImage/libAfterImage
 * NAME
 * libAfterImage - generic imaging library originally designed for 
 * AfterStep X Window Manager.
 *
 * PURPOSE
 * libAfterImage is the imaging library implemented for AfterStep
 * X Window Manager. It has been generalized to be suitable for any
 * application in need of robust graphics engine.
 *
 * It provides facilities for loading images from files of different
 * formats, compressed in memory storage of images, scaling,
 * tinting/shading, flipping and superimposition of arbitrary number of
 * images over each other. In addition it allows for linear gradients
 * drawing, and antialiased/smoothed text drawing using both  FreeType
 * library and X Window fonts.
 *
 * Primary goals of this library are to achieve exceptional quality of
 * images and text, making code fast and small at the same time.
 * Additional steps are taken to compensate for screen colordepth
 * limitation, and different error diffusion algorithms are used to
 * provide for smooth images even in low colordepth modes.
 *
 * HISTORY
 * libAfterImage has been implemented as an attempt to address several
 * issues. First one, and most important is that X Window System
 * completely lacks any tools for image manipulation, except for very
 * rudimentary operations. With Introduction of Render extentions in
 * XFree86 4.0 that situation is changing, but still is far from
 * perfect. There need is therefore to implement client side image
 * manipulation. That inturn creates a problem of image data transfer
 * between Server and client.
 *
 * To avoid that full-time image storage on the client side is needed.
 * Which is memory extensive. So there appears to be a need for some
 * in-memory compression.
 *
 * On the other side there is an image quality issue. Its easy to write
 * a scaling function by simply dropping out or duplicating pixels,
 * but quality is awfull. What is needed is very fast
 * averaging/interpolation code. That brings us to the issue of 8 bits
 * per channel. Add two pixels together and you get overflow. So all the
 * math has to be performed in different colorspace and then converted
 * back. On the other side, such a conversion may discard valuable bits,
 * so some compensation method has to be implemented.
 *
 * On the text drawing side of things, there are numerous problems just
 * as well. Native X fonts look ugly as soon as you try to show any
 * decently sized text. That is supposed to be solved with said Render
 * extensions to XFree86, but as experiense has shown, things aren't as
 * easy as it looks, besides one wants his app to run well under any X
 * Server. FreeType library provides a solution, but not always
 * available. Another problem is that if you keep all your images on the
 * client side, you want to draw text on client side as well.
 *
 * The solution is to provide transparent interface that could use both
 * X fonts and FreeType, cache glyphs on the client side and possibly
 * perform smoothing of ugly X fonts.
 *
 * There is no library solving all this problems in one fell swoop,
 * except for monstrous ones, like ImLib.
 *
 * Hence libAfterImage has come to life.
 *
 * DESCRIPTION
 * libAfterStep provides sevarl facilities.
 *
 * 1. X Visual abstruction layer via ASVisual. This layer handles color
 * management, transfer of data to and from X Server, and other screen
 * related stuff.
 *
 * 2. Scanline handling facility via ASScanline. ASScanline is the most
 * widely used structure since image handling is implemented on
 * per-scanline basis.
 *
 * 3. Image storage, trasformation and rendering via ASImage. ASImage
 * provides for generic container used for any image or text
 * manipulation. It incorporates such a robust facilities as in-memory
 * RLE compression, separate channel storage of 4 channels ( Alpha, Red,
 * Green, and Blue ) with 8 bit per channel.
 *
 * 4. Simplified font handling facility via ASFont and ASFointManager.
 * All the font handling is done using ASFont structure, no matter what
 * type of font is used. Any font supported by X11 and FreeType can be
 * used.
 *
 * 5. Transparent image file reading for many different formats. Included
 * built-in XPM reading code and XCF ( GIMP's native format ).
 * Overall supported:
 * via external libraries :
 * XPM, JPEG, PNG, TIFF, GIF
 * via built in code :
 * XPM, XCF, PNM, PPM, BMP, ICO, CUR
 * Note that XPM can be supported both via built-in code or via libXpm,
 * depending on compilation configuration.
 * Actuall image file format is autodetected from the file contents -
 * file name extention is not used and can be anything at all.
 *
 * 6. Image export into many popular file formats. Currently implemented :
 * XPM, JPEG, PNG, GIF. Work is underway to implement support for TIFF,
 * XCF, BMP, ICO.
 *
 * 7. Image quantization to arbitrary size colormap.
 *
 * 8. libAfterImage could be used without X window system, which is
 * coninient for such thing as web development. XML Image manipulation
 * tool, that could be used in such activity is included (see ascompose.c)
 *
 * 9. Image reference counting
 *
 * USES
 * libAfterBase - AfterStep basic functionality library. That Includes
 * Hash tables, file search methods, message output, generic types.
 * However effort has been made to allow for standalone configuration as
 * well. If libAfterBase is not found at compilation time - libAfterImage
 * will use extract from libAfterBase included with libAfterImage.
 *
 * SEE ALSO
 * Examples
 * API Reference
 *
 * TODO
 * Implement support for Targa and PCX image format and maybe some other
 * formats as well.
 *
 * Implement complete support for I18N internationalization.
 *
 * Implement color<->pixel conversion for all colordepths.
 *
 * AUTHOR
 * Sasha Vasko <sasha at aftercode dot net>
 *********/
/****h* libAfterImage/Examples
 * EXAMPLE
 * ASView  - image loading from the file and displaying in window.
 * ASScale - image loading from file and scaling to arbitrary size.
 * ASTile  - image loading from file, tiling and tinting to arbitrary
 *           size and color.
 * ASMerge - imgae loading and merging with another image.
 * ASGrad  - mutlipoint gradient drawing.
 * ASFlip  - image loading from file and rotation.
 * ASText  - trexturized semitransparent antialised text drawing.
 *
 * SEE ALSO
 * API Reference
 ******/
/****h* libAfterImage/API Reference
 * CHILDREN
 * Headers :
 *          ascmap.h asfont.h asimage.h asvisual.h blender.h export.h
 *          import.h transform.h ximage.h
 * Structures :
 *          ColorPair
 *          ASScanline
 *          ASVisual
 *          ASImage
 *          ASImageManager
 *          ASImageBevel
 *          ASImageDecoder
 *          ASImageOutput
 *          ASImageLayer
 *          ASGradient
 *          ASFontManager
 *          ASFont
 *          ASGlyph
 *          ASGlyphRange
 *          ASColormap
 *          ASImageExportParams
 *          ASVectorPalette
 *
 * Functions :
 *   ASScanline handling:
 *  	    prepare_scanline(), free_scanline()
 *
 *   ASVisual initialization :
 *  	    query_screen_visual(), setup_truecolor_visual(),
 *  	    setup_pseudo_visual(), setup_as_colormap(),
 *          create_asvisual(), create_asvisual_for_id(),
 *  	    destroy_asvisual()
 *
 *   ASVisual encoding/decoding :
 *  	    visual2visual_prop(), visual_prop2visual()
 *
 *   ASVisual convenience functions :
 *  	    create_visual_window(), create_visual_pixmap(),
 *  	    create_visual_ximage(), create_visual_gc()
 *
 *   Colorspace conversion :
 *          rgb2value(), rgb2saturation(), rgb2hue(), rgb2luminance(),
 *          rgb2hsv(), rgb2hls(), hsv2rgb(), hls2rgb(),
 *          degrees2hue16(), hue162degrees(), normalize_degrees_val()
 *
 *   Image quantization :
 *          colormap_asimage(), destroy_colormap()
 *
 *   merge_scanline methods :
 *          alphablend_scanlines(), allanon_scanlines(),
 *          tint_scanlines(), add_scanlines(), sub_scanlines(),
 *          diff_scanlines(), darken_scanlines(), lighten_scanlines(),
 *          screen_scanlines(), overlay_scanlines(), hue_scanlines(),
 *          saturate_scanlines(), value_scanlines(),
 *          colorize_scanlines(), dissipate_scanlines().
 *
 *   ASImage handling :
 *          asimage_init(), asimage_start(), create_asimage(),
 *          clone_asimage(), destroy_asimage()
 *
 *   ASImage channel data manipulations :
 *          get_asimage_chanmask(), move_asimage_channel(),
 *          copy_asimage_channel(), copy_asimage_lines()
 *
 *   ImageManager Reference counting and managing :
 *          create_image_manager(), destroy_image_manager(),
 *          store_asimage(), fetch_asimage(), dup_asimage(),
 *          release_asimage(), release_asimage_by_name()
 *
 *   Layers helper functions :
 *          init_image_layers(), create_image_layers(),
 *          destroy_image_layers()
 *
 *   Encoding :
 *          asimage_add_line(),	asimage_add_line_mono(),
 *          asimage_print_line()
 *
 *   Decoding :
 *          start_image_decoding(), stop_image_decoding(),
 *          asimage_decode_line (), set_decoder_shift(),
 *          set_decoder_bevel_geom(), set_decoder_back_color()
 *
 *   ASImage from scientific data :
 *          set_asimage_vector(), colorize_asimage_vector(),
 *          create_asimage_from_vector()
 *
 *   Output :
 *          start_image_output(), set_image_output_back_color(),
 *          toggle_image_output_direction(), stop_image_output()
 *
 *   X11 conversions :
 *          ximage2asimage(), pixmap2asimage(), asimage2ximage(),
 *          asimage2mask_ximage(), asimage2pixmap(), asimage2mask()
 *
 *   Transformations :
 *          scale_asimage(), tile_asimage(), merge_layers(), 
 * 			make_gradient(),
 *          flip_asimage(), mirror_asimage(), pad_asimage(),
 *          blur_asimage_gauss(), fill_asimage(), adjust_asimage_hsv()
 *
 *   Import :
 *          file2ASImage(), file2pixmap()
 *   Export :
 *   		ASImage2file()
 *
 *   Text Drawing :
 *          create_font_manager(), destroy_font_manager(),
 *          open_freetype_font(), open_X11_font(), get_asfont(),
 *          release_font(), print_asfont(), print_asglyph(),
 *          draw_text(), draw_fancy_text()
 *********/
#endif  /* AFTERIMAGE_H_HEADER_INCLUDED */

