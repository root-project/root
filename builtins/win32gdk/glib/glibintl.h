#ifndef __GLIBINTL_H__
#define __GLIBINTL_H__

#include "config.h"

#ifdef ENABLE_NLS

gchar *_glib_gettext (const gchar *str);

#include <libintl.h>
#define _(String) _glib_gettext(String)

#ifdef gettext_noop
#define N_(String) gettext_noop(String)
#else
#define N_(String) (String)
#endif
#else /* NLS is disabled */
#define _(String) (String)
#define N_(String) (String)
#define textdomain(String) (String)
#define gettext(String) (String)
#define dgettext(Domain,String) (String)
#define dcgettext(Domain,String,Type) (String)
#define bindtextdomain(Domain,Directory) (Domain) 
#endif

#endif /* __GLIBINTL_H__ */
