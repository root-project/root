/* jconfig.h.  adapted to use uper level config.h from libAfterImage.  */

#ifdef _WIN32
#include "../win32/config.h"
#include <basetsd.h>
#else
#include "../config.h"
#endif

#ifdef __CHAR_UNSIGNED__
# define CHAR_IS_UNSIGNED
#endif

#ifdef HAVE_STRINGS_H
# define NEED_BSD_STRINGS
#endif

#undef NEED_FAR_POINTERS
#undef NEED_SHORT_EXTERNAL_NAMES
/* Define this if you get warnings about undefined structures. */
#undef INCOMPLETE_TYPES_BROKEN

#ifdef JPEG_INTERNALS

#undef RIGHT_SHIFT_IS_UNSIGNED
/* These are for configuring the JPEG memory manager. */
#undef DEFAULT_MAX_MEM
#undef NO_MKTEMP

#endif /* JPEG_INTERNALS */

