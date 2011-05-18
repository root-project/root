/*
 * Copyright (C) 2001 Sasha Vasko <sasha at aftercode.net>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 */
#ifdef _WIN32
#include "win32/config.h"
#else
#include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <limits.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_STDLIB_H
#include <stdlib.h>
#endif
#ifdef HAVE_STDDEF_H
#include <stddef.h>
#endif
#ifdef HAVE_STDARG_H
#include <stdarg.h>
#endif
#include <sys/stat.h>
#ifdef __APPLE_API_PRIVATE
#undef __APPLE_API_PRIVATE
#endif
#if HAVE_DIRENT_H
# include <dirent.h>
# define NAMLEN(dirent) strlen((dirent)->d_name)
#else
# if HAVE_SYS_DIRENT_H
#  include <sys/dirent.h>
#  define NAMLEN(dirent) strlen((dirent)->d_name)
# else
#  define dirent direct
#  define NAMLEN(dirent) (dirent)->d_namlen
#  if HAVE_SYS_NDIR_H
#   include <sys/ndir.h>
#  endif
#  if HAVE_SYS_DIR_H
#   include <sys/dir.h>
#  endif
#  if HAVE_NDIR_H
#   include <ndir.h>
#  endif
# endif
#endif

#ifdef _WIN32
#include "win32/afterbase.h"
#include <io.h>
#include <windows.h>
#define access _access
#else
#include "afterbase.h"
#endif
#include "asimage.h"

#ifdef X_DISPLAY_MISSING
#include "colornames.h"
#endif

/*#include <X11/Xlib.h>*/

char    *asim_ApplicationName = NULL ;

void
asim_set_application_name (char *argv0)
{
	char *temp = &(argv0[0]);
	do
	{	/* Save our program name - for error messages */
		register int i = 1 ;                   /* we don't use standard strrchr since there
												* seems to be some wierdness in
												* CYGWIN implementation of it. */
		asim_ApplicationName =  temp ;
		while( temp[i] && temp[i] != '/' ) ++i ;
		temp = temp[i] ? &(temp[i+1]): NULL ;
	}while( temp != NULL );
}

const char *
asim_get_application_name()
{
	return asim_ApplicationName;
}

static unsigned int asim_as_output_threshold = OUTPUT_DEFAULT_THRESHOLD ;

unsigned int
asim_get_output_threshold()
{
	return asim_as_output_threshold ;
}

unsigned int
asim_set_output_threshold( unsigned int threshold )
{
    unsigned int old = asim_as_output_threshold;
    asim_as_output_threshold = threshold ;
    return old;
}


/* from libAfterBase/output.c : */
Bool asim_show_error( const char *error_format, ...)
{
    if( OUTPUT_LEVEL_ERROR <= get_output_threshold())
    {
        va_list ap;
        fprintf (stderr, "%s ERROR: ", get_application_name() );
        va_start (ap, error_format);
        vfprintf (stderr, error_format, ap);
        va_end (ap);
        fprintf (stderr, "\n" );
        return True;
    }
    return False;
}

Bool asim_show_warning( const char *warning_format, ...)
{
    if( OUTPUT_LEVEL_WARNING <= get_output_threshold())
    {
        va_list ap;
        fprintf (stderr, "%s warning: ", get_application_name() );
        va_start (ap, warning_format);
        vfprintf (stderr, warning_format, ap);
        va_end (ap);
        fprintf (stderr, "\n" );
        return True;
    }
    return False;
}

Bool asim_show_progress( const char *msg_format, ...)
{
    if( OUTPUT_LEVEL_PROGRESS <= get_output_threshold())
    {
        va_list ap;
        fprintf (stderr, "%s : ", get_application_name() );
        va_start (ap, msg_format);
        vfprintf (stderr, msg_format, ap);
        va_end (ap);
        fprintf (stderr, "\n" );
        return True;
    }
    return False;
}


Bool asim_show_debug( const char *file, const char *func, int line, const char *msg_format, ...)
{
    if( OUTPUT_LEVEL_DEBUG <= get_output_threshold())
    {
        va_list ap;
        fprintf (stderr, "%s debug msg: %s:%s():%d: ", get_application_name(), file, func, line );
        va_start (ap, msg_format);
        vfprintf (stderr, msg_format, ap);
        va_end (ap);
        fprintf (stderr, "\n" );
        return True;
    }
    return False;
}


void asim_nonGNUC_debugout( const char *format, ...)
{
    va_list ap;
    fprintf (stderr, "%s: ", get_application_name() );
    va_start (ap, format);
    vfprintf (stderr, format, ap);
    va_end (ap);
    fprintf (stderr, "\n" );
}

void asim_nonGNUC_debugout_stub( const char *format, ...)
{}

/* from libAfterBase/fs.c : */

int		asim_check_file_mode (const char *file, int mode)
{
	struct stat   st;

	if ((stat (file, &st) == -1) || (st.st_mode & S_IFMT) != mode)
		return (-1);
	else
		return (0);
}

char         *
asim_put_file_home (const char *path_with_home)
{
	static char  *home = NULL;				   /* the HOME environment variable */
	static char   default_home[3] = "./";
	static int    home_len = 0;
	char         *str = NULL, *ptr;
	register int  i;
	if (path_with_home == NULL)
		return NULL;
	/* home dir ? */
	if ( strncmp(  path_with_home, "$HOME/", 6 ) == 0 )
		path_with_home += 5 ;
	else if (path_with_home[0] == '~' && path_with_home[1] == '/')
		path_with_home += 1 ;
	else
		return mystrdup(path_with_home);

	if (home == NULL)
	{
		if ((home = getenv ("HOME")) == NULL)
			home = &(default_home[0]);
		home_len = strlen (home);
	}

	for (i = 0; path_with_home[i]; i++);
	str = safemalloc (home_len + i + 1);
	for (ptr = str + home_len; i >= 0; i--)
		ptr[i] = path_with_home[i];
	for (i = 0; i < home_len; i++)
		str[i] = home[i];
	return str;
}

char*
asim_load_binary_file(const char* realfilename, long *file_size_return)
{
	struct stat st;
	FILE* fp;
	char* data = NULL ;

	/* Get the file size. */
	if (stat(realfilename, &st)) return NULL;
	/* Open the file. */
	fp = fopen(realfilename, "rb");
	if ( fp != NULL ) 
	{
		long len ; 
		/* Read in the file. */
		data = safecalloc(1, st.st_size + 1);
		len = fread(data, 1, st.st_size, fp);
		if( file_size_return ) 
			*file_size_return = len ; 
		fclose(fp);
	}
	return data;
}

char*
asim_load_file(const char* realfilename)
{
	long len;
	char* str = load_binary_file( realfilename, &len );

	if (str != NULL && len >= 0) 
		str[len] = '\0';

	return str;
}


/****************************************************************************
 *
 * Find the specified icon file somewhere along the given path.
 *
 * There is a possible race condition here:  We check the file and later
 * do something with it.  By then, the file might not be accessible.
 * Oh well.
 *
 ****************************************************************************/
/* supposedly pathlist should not include any environment variables
   including things like ~/
 */
void 
unix_path2dos_path( char *path )
{
	int i = strlen(path) ; 
	while( --i >= 0 ) 
		if( path[i] == '/' && ( i == 0 || path[i-1] != '/' ) )
			path[i] = '\\' ;
}		   


char         *
asim_find_file (const char *file, const char *pathlist, int type)
{
	char 		  *path;
	register int   len;
	int            max_path = 0;
	register char *ptr;
	register int   i;
	Bool local = False ;

	if (file == NULL)
		return NULL;
#ifdef _WIN32
#define PATH_SEPARATOR_CHAR ';'
#define PATH_CHAR '\\'
#else
#define PATH_SEPARATOR_CHAR ':'
#define PATH_CHAR '/'
#endif
	
	if (*file == PATH_CHAR || *file == '~' || ((pathlist == NULL) || (*pathlist == '\0')))
		local = True ;
	else if( file[0] == '.' && (file[1] == PATH_CHAR || (file[1] == '.' && file[2] == PATH_CHAR))) 
		local = True ;
	else if( strncmp( file, "$HOME", 5) == 0 ) 
		local = True ;
	if( local ) 
	{
		path = put_file_home (file);
		if ( access (path, type) == 0 )
		{
			return path;
		}
		free (path);
		return NULL;
	}
/*	return put_file_home(file); */
	for (i = 0; file[i]; i++);
	len = i ;
	for (ptr = (char *)pathlist; *ptr; ptr += i)
	{
		if (*ptr == PATH_SEPARATOR_CHAR )
			ptr++;
		for (i = 0; ptr[i] && ptr[i] != PATH_SEPARATOR_CHAR; i++);
		if (i > max_path)
			max_path = i;
	}

	path = safecalloc (1, max_path + 1 + len + 1);
	strcpy( path+max_path+1, file );
	path[max_path] = PATH_CHAR ;

	ptr = (char*)&(pathlist[0]) ;
	while( ptr[0] != '\0' )
	{
		int skip ;
		for( i = 0 ; ptr[i] == PATH_SEPARATOR_CHAR; ++i );
		ptr += i ;
		for( i = 0 ; ptr[i] != PATH_SEPARATOR_CHAR && ptr[i] != '\0'; ++i );
		skip = i ;
		if( i > 0 && ptr[i-1] == PATH_CHAR )
			i-- ;
		if( i > 0 )
		{
			register char *try_path = path+max_path-i;
			strncpy( try_path, ptr, i );
			if (access(try_path, type) == 0)
			{
				char* res = mystrdup(try_path);
				free( path );
				return res;
			}
		}
		ptr += skip ;
	}
	free (path);
	return NULL;
}

static char         *
find_envvar (char *var_start, int *end_pos)
{
	char          backup, *name_start = var_start;
	register int  i;
	char         *var = NULL;

	if (var_start[0] == '{')
	{
		name_start++;
		for (i = 1; var_start[i] && var_start[i] != '}'; i++);
	} else
		for (i = 0; isalnum ((int)var_start[i]) || var_start[i] == '_'; i++);

	backup = var_start[i];
	var_start[i] = '\0';
	var = getenv (name_start);
	var_start[i] = backup;

	*end_pos = i;
	if (backup == '}')
		(*end_pos)++;
	return var;
}

static char *
do_replace_envvar (char *path)
{
	char         *data = path, *tmp;
	char         *home = getenv ("HOME");
	int           pos = 0, len, home_len = 0;

	if (path == NULL)
		return NULL;
	if (*path == '\0')
		return path;
	len = strlen (path);
	if (home)
		home_len = strlen (home);

	while (*(data + pos))
	{
		char         *var;
		int           var_len, end_pos;

		while (*(data + pos) != '$' && *(data + pos))
		{
			if (*(data + pos) == '~' && *(data + pos + 1) == '/')
			{
				if (pos > 0)
					if (*(data + pos - 1) != ':')
					{
						pos += 2;
						continue;
					}
				if (home == NULL)
					*(data + (pos++)) = '.';
				else
				{
					len += home_len;
					tmp = safecalloc (1, len);
					strncpy (tmp, data, pos);
					strcpy (tmp + pos, home);
					strcpy (tmp + pos + home_len, data + pos + 1);
					if( data != path )
						free (data);
					data = tmp;
					pos += home_len;
				}
			}
			pos++;
		}
		if (*(data + pos) == '\0')
			break;
		/* found $ sign - trying to replace var */
		if ((var = find_envvar (data + pos + 1, &end_pos)) == NULL)
		{
			++pos;
			continue;
		}
		var_len = strlen (var);
		len += var_len;
		tmp = safecalloc (1, len);
		strncpy (tmp, data, pos);
		strcpy (tmp + pos, var);
		strcpy (tmp + pos + var_len, data + pos + end_pos + 1);
		if( data != path )
			free (data);
		data = tmp;
	}
	return data;
}

char*
asim_copy_replace_envvar (char *path)
{
	char         *res = do_replace_envvar( path );
	return ( res == path )?mystrdup( res ):res;
}


/*******************************************************************/
/* from mystring.c : */
char         *
asim_mystrndup (const char *str, size_t n)
{
	char         *c = NULL;
	if (str)
	{
		c = calloc (1, n + 1);
		strncpy (c, str, n);
	}
	return c;
}

char         *
asim_mystrdup (const char *str)
{
	char         *c = NULL;

	if (str)
	{
		c = malloc (strlen (str) + 1);
		strcpy (c, str);
	}
	return c;
}

int
asim_mystrcasecmp (const char *s1, const char *s2)
{
	int          c1, c2;
	register int i = 0 ;

	if (s1 == NULL || s2 == NULL)
		return (s1 == s2) ? 0 : ((s1==NULL)?1:-1);
	while (s1[i])
	{
		/* in some BSD implementations, tolower(c) is not defined
		 * unless isupper(c) is true */
		c1 = s1[i];
		if (isupper (c1))
			c1 = tolower (c1);
		c2 = s2[i];
		if (isupper (c2))
			c2 = tolower (c2);

		++i ;
		if (c1 != c2)
			return (c1 - c2);
	}
	return -s2[i];
}

int
asim_mystrncasecmp (const char *s1, const char *s2, size_t n)
{
	register int  c1, c2;
	register int i = 0 ;

	if (s1 == NULL || s2 == NULL)
		return (s1 == s2) ? 0 : ((s1==NULL)?1:-1);
	while( i < n )
	{
		c1 = s1[i], c2 = s2[i];
		++i ;
		if (c1==0)
			return -c2;
		if (isupper (c1))
			c1 = tolower(c1);
		if (isupper (c2))
			c2 = tolower(c2);
		if (c1 != c2)
			return (c1 - c2);
	}
	return 0;
}

#ifdef X_DISPLAY_MISSING
static int compare_xcolor_entries(const void *a, const void *b)
{
   return strcmp((const char *) a, ((const XColorEntry *) b)->name);
}

static int FindColor(const char *name, CARD32 *colorPtr)
{
   XColorEntry *found;

   found = bsearch(name, xColors, numXColors, sizeof(XColorEntry),
                   compare_xcolor_entries);
   if (found == NULL)
      return 0;

   *colorPtr = 0xFF000000|((found->red<<16)&0x00FF0000)|((found->green<<8)&0x0000FF00)|((found->blue)&0x000000FF);
   return 1;
}
#endif

/*******************************************************************/
/* from parse,c : */
const char *asim_parse_argb_color( const char *color, CARD32 *pargb )
{
#define hextoi(h)   (isdigit(h)?((h)-'0'):(isupper(h)?((h)-'A'+10):((h)-'a'+10)))
	if( color )
	{
		if( *color == '#' )
		{
			CARD32 argb = 0 ;
			int len = 0 ;
			register const char *ptr = color+1 ;
			while( isxdigit((int)ptr[len]) ) len++;
			if( len >= 3)
			{
				if( (len&0x3) == 0 && len != 12 )
				{  /* we do have alpha channel !!! */
					len = len>>2 ;
					argb = (hextoi((int)ptr[0])<<28)&0xF0000000 ;
					if( len > 1 )
						argb |= (hextoi((int)ptr[1])<<24)&0x0F000000 ;
					else
						argb |= 0x0F000000;
					ptr += len ;
				}else
				{
					len = len/3 ;
					argb = 0xFF000000;
				}
				/* processing rest of the channels : */
				if( len == 1 )
				{
					argb |= 0x000F0F0F;
					argb |= (hextoi((int)ptr[0])<<20)&0x00F00000 ;
					argb |= (hextoi((int)ptr[1])<<12)&0x0000F000 ;
					argb |= (hextoi((int)ptr[2])<<4 )&0x000000F0 ;
					ptr += 3 ;
				}else
				{
					argb |= (hextoi((int)ptr[0])<<20)&0x00F00000 ;
					argb |= (hextoi((int)ptr[1])<<16)&0x000F0000 ;
					ptr += len ;
					argb |= (hextoi((int)ptr[0])<<12)&0x0000F000 ;
					argb |= (hextoi((int)ptr[1])<<8) &0x00000F00 ;
					ptr += len ;
					argb |= (hextoi((int)ptr[0])<<4 )&0x000000F0 ;
					argb |= (hextoi((int)ptr[1]))    &0x0000000F ;
					ptr += len ;
				}
				*pargb = argb ;
				return ptr;
			}
		}else if( *color )
		{
			/* does not really matter here what screen to use : */
			Display *dpy = get_default_asvisual()->dpy;
#ifdef X_DISPLAY_MISSING
			register const char *ptr = &(color[0]);
            if(!FindColor(color, pargb))
                return color;
    		while( !isspace((int)*ptr) && *ptr != '\0' ) ptr++;
			return ptr;
#else
			if( dpy == NULL )
				return color ;
			else
			{
				register const char *ptr = &(color[0]);
#ifndef X_DISPLAY_MISSING
				XColor xcol, xcol_scr ;
/* XXX Not sure if Scr.asv->colormap is always defined here.  If not,
** change back to DefaultColormap(dpy,DefaultScreen(dpy)). */
				if( XLookupColor( dpy, DefaultColormap(dpy,DefaultScreen(dpy)), color, &xcol, &xcol_scr) )
					*pargb = 0xFF000000|((xcol.red<<8)&0x00FF0000)|(xcol.green&0x0000FF00)|((xcol.blue>>8)&0x000000FF);
#endif
				while( !isspace((int)*ptr) && *ptr != '\0' ) ptr++;
				return ptr;
			}
#endif
		}
	}
	return color;
}


static int asim_asxml_var_nget(char* name, int n);


/* Math expression parsing algorithm. */
double asim_parse_math(const char* str, char** endptr, double size) {
	double total = 0;
	char op = '+';
	char minus = 0;
	char logical_not = 0;
/*	const char* startptr = str; */
	if( str == NULL ) 
		return 0 ;

	while (isspace((int)*str)) str++;
	if( *str == '!' ) 
	{
		logical_not = 1;
		++str ;
	}else if( *str == '-' ) 
	{
		minus = 1 ;
		++str ;
	}

	while (*str) 
	{
		while (isspace((int)*str)) str++;
		if (!op) 
		{
			if (*str == '+' || *str == '-' || *str == '*' || *str == '/') op = *str++;
			else if (*str == '-') { minus = 1; str++; }
			else if (*str == '!') { logical_not = 1; str++; }
			else if (*str == ')') { str++; break; }
			else break;
		} else 
		{
			char* ptr;
			double num;
			
			if (*str == '(') 
				num = asim_parse_math(str + 1, &ptr, size);
            else if (*str == '$') 
			{
            	for (ptr = (char*)str + 1 ; *ptr && !isspace(*ptr) && *ptr != '+' && *ptr != '-' && *ptr != '*' && *ptr != '!' && *ptr != '/' && *ptr != ')' ; ptr++);
               	num = asim_asxml_var_nget((char*)str + 1, ptr - (str + 1));
            }else 
				num = strtod(str, &ptr);
			
			if (str != ptr) 
			{
				if (*ptr == '%') num *= size / 100.0, ptr++;
				if (minus) num = -num;
				if (logical_not) num = !num;
				
				if (op == '+') total += num;
				else if (op == '-') total -= num;
				else if (op == '*') total *= num;
				else if (op == '/' && num) total /= num;
			} else 
				break;
			str = ptr;
			op = '\0';
			minus = logical_not = 0;
		}
	}
	if (endptr) *endptr = (char*)str;
/* 	show_debug(__FILE__,"parse_math",__LINE__,"Parsed math [%s] with reference [%.2f] into number [%.2f].", startptr, size, total); */
	return total;
}

/*******************************************************************/
/* from ashash,c : */
ASHashKey asim_default_hash_func (ASHashableValue value, ASHashKey hash_size)
{
	return (ASHashKey)(value % hash_size);
}

long
asim_default_compare_func (ASHashableValue value1, ASHashableValue value2)
{
	return ((long)value1 - (long)value2);
}

long
asim_desc_long_compare_func (ASHashableValue value1, ASHashableValue value2)
{
    return ((long)value2 - (long)value1);
}

void
asim_init_ashash (ASHashTable * hash, Bool freeresources)
{
LOCAL_DEBUG_CALLER_OUT( " has = %p, free ? %d", hash, freeresources );
	if (hash)
	{
		if (freeresources)
			if (hash->buckets)
				free (hash->buckets);
		memset (hash, 0x00, sizeof (ASHashTable));
	}
}

ASHashTable  *
asim_create_ashash (ASHashKey size,
			   ASHashKey (*hash_func) (ASHashableValue, ASHashKey),
			   long (*compare_func) (ASHashableValue, ASHashableValue),
			   void (*item_destroy_func) (ASHashableValue, void *))
{
	ASHashTable  *hash;

	if (size <= 0)
		size = 63;

	hash = safecalloc (1, sizeof (ASHashTable));
	init_ashash (hash, False);

	hash->buckets = safecalloc (size, sizeof (ASHashBucket));

	hash->size = size;

	if (hash_func)
		hash->hash_func = hash_func;
	else
		hash->hash_func = asim_default_hash_func;

	if (compare_func)
		hash->compare_func = compare_func;
	else
		hash->compare_func = asim_default_compare_func;

	hash->item_destroy_func = item_destroy_func;

	return hash;
}

static void
destroy_ashash_bucket (ASHashBucket * bucket, void (*item_destroy_func) (ASHashableValue, void *))
{
	register ASHashItem *item, *next;

	for (item = *bucket; item != NULL; item = next)
	{
		next = item->next;
		if (item_destroy_func)
			item_destroy_func (item->value, item->data);
		free (item);
	}
	*bucket = NULL;
}

void
asim_destroy_ashash (ASHashTable ** hash)
{
LOCAL_DEBUG_CALLER_OUT( " hash = %p, *hash = %p", hash, *hash  );
	if (*hash)
	{
		register int  i;

		for (i = (*hash)->size - 1; i >= 0; i--)
			if ((*hash)->buckets[i])
				destroy_ashash_bucket (&((*hash)->buckets[i]), (*hash)->item_destroy_func);

		asim_init_ashash (*hash, True);
		free (*hash);
		*hash = NULL;
	}
}

static        ASHashResult
add_item_to_bucket (ASHashBucket * bucket, ASHashItem * item, long (*compare_func) (ASHashableValue, ASHashableValue))
{
	ASHashItem  **tmp;

	/* first check if we already have this item */
	for (tmp = bucket; *tmp != NULL; tmp = &((*tmp)->next))
	{
		register long res = compare_func ((*tmp)->value, item->value);

		if (res == 0)
			return ((*tmp)->data == item->data) ? ASH_ItemExistsSame : ASH_ItemExistsDiffer;
		else if (res > 0)
			break;
	}
	/* now actually add this item */
	item->next = (*tmp);
	*tmp = item;
	return ASH_Success;
}

#define DEALLOC_CACHE_SIZE      1024
static ASHashItem*  deallocated_mem[DEALLOC_CACHE_SIZE+10] ;
static unsigned int deallocated_used = 0 ;

ASHashResult
asim_add_hash_item (ASHashTable * hash, ASHashableValue value, void *data)
{
	ASHashKey     key;
	ASHashItem   *item;
	ASHashResult  res;

	if (hash == NULL)
        return ASH_BadParameter;

	key = hash->hash_func (value, hash->size);
	if (key >= hash->size)
        return ASH_BadParameter;

    if( deallocated_used > 0 )
        item = deallocated_mem[--deallocated_used];
    else
        item = safecalloc (1, sizeof (ASHashItem));

	item->next = NULL;
	item->value = value;
	item->data = data;

	res = add_item_to_bucket (&(hash->buckets[key]), item, hash->compare_func);
	if (res == ASH_Success)
	{
		hash->most_recent = item ;
		hash->items_num++;
		if (hash->buckets[key]->next == NULL)
			hash->buckets_used++;
	} else
		free (item);
	return res;
}

static ASHashItem **
find_item_in_bucket (ASHashBucket * bucket,
					 ASHashableValue value, long (*compare_func) (ASHashableValue, ASHashableValue))
{
	register ASHashItem **tmp;
	register long res;

	/* first check if we already have this item */
	for (tmp = bucket; *tmp != NULL; tmp = &((*tmp)->next))
	{
		res = compare_func ((*tmp)->value, value);
		if (res == 0)
			return tmp;
		else if (res > 0)
			break;
	}
	return NULL;
}

ASHashResult
asim_get_hash_item (ASHashTable * hash, ASHashableValue value, void **trg)
{
	ASHashKey     key;
	ASHashItem  **pitem = NULL;

	if (hash)
	{
		key = hash->hash_func (value, hash->size);
		if (key < hash->size)
			pitem = find_item_in_bucket (&(hash->buckets[key]), value, hash->compare_func);
	}
	if (pitem)
		if (*pitem)
		{
			if (trg)
				*trg = (*pitem)->data;
			return ASH_Success;
		}
	return ASH_ItemNotExists;
}

ASHashResult
asim_remove_hash_item (ASHashTable * hash, ASHashableValue value, void **trg, Bool destroy)
{
	ASHashKey     key = 0;
	ASHashItem  **pitem = NULL;

	if (hash)
	{
		key = hash->hash_func (value, hash->size);
		if (key < hash->size)
			pitem = find_item_in_bucket (&(hash->buckets[key]), value, hash->compare_func);
	}
	if (pitem)
		if (*pitem)
		{
			ASHashItem   *next;

			if( hash->most_recent == *pitem )
				hash->most_recent = NULL ;

			if (trg)
				*trg = (*pitem)->data;

			next = (*pitem)->next;
			if (hash->item_destroy_func && destroy)
				hash->item_destroy_func ((*pitem)->value, (trg) ? NULL : (*pitem)->data);

            if( deallocated_used < DEALLOC_CACHE_SIZE )
            {
                deallocated_mem[deallocated_used++] = *pitem ;
            }else
                free( *pitem );

            *pitem = next;
			if (hash->buckets[key] == NULL)
				hash->buckets_used--;
			hash->items_num--;

			return ASH_Success;
		}
	return ASH_ItemNotExists;
}

void asim_flush_ashash_memory_pool()
{
	/* we better disable errors as some of this data will belong to memory audit : */
	while( deallocated_used > 0 )
		free( deallocated_mem[--deallocated_used] );
}

/************************************************************************/
/************************************************************************/
/* 	Some usefull implementations 					*/
/************************************************************************/
ASHashKey asim_pointer_hash_value (ASHashableValue value, ASHashKey hash_size)
{
    union
    {
        void *ptr;
        ASHashKey key[2];
    } mix;
    register  ASHashKey key;

    mix.ptr = (void*)value;
    key = mix.key[0]^mix.key[1] ;
    if( hash_size == 256 )
		return (key>>4)&0x0FF;
    return (key>>4) % hash_size;
}

/* case sensitive strings hash */
ASHashKey
asim_string_hash_value (ASHashableValue value, ASHashKey hash_size)
{
	ASHashKey     hash_key = 0;
	register int  i = 0;
	char         *string = (char*)value;
	register char c;

	do
	{
		c = string[i];
		if (c == '\0')
			break;
		hash_key += (((ASHashKey) c) << i);
		++i ;
	}while( i < ((sizeof (ASHashKey) - sizeof (char)) << 3) );
	return hash_key % hash_size;
}

long
asim_string_compare (ASHashableValue value1, ASHashableValue value2)
{
	register char *str1 = (char*)value1;
	register char *str2 = (char*)value2;
	register int   i = 0 ;

	if (str1 == str2)
		return 0;
	if (str1 == NULL)
		return -1;
	if (str2 == NULL)
		return 1;
	do
	{
		if (str1[i] != str2[i])
			return (long)(str1[i]) - (long)(str2[i]);

	}while( str1[i++] );
	return 0;
}

void
asim_string_destroy_without_data (ASHashableValue value, void *data)
{
	if ((char*)value != NULL)
		free ((char*)value);
}

/* variation for case-unsensitive strings */
ASHashKey
asim_casestring_hash_value (ASHashableValue value, ASHashKey hash_size)
{
	ASHashKey     hash_key = 0;
	register int  i = 0;
	char         *string = (char*)value;
	register int c;

	do
	{
		c = string[i];
		if (c == '\0')
			break;
		if (isupper (c))
			c = tolower (c);
		hash_key += (((ASHashKey) c) << i);
		++i;
	}while(i < ((sizeof (ASHashKey) - sizeof (char)) << 3));

	return hash_key % hash_size;
}

long
asim_casestring_compare (ASHashableValue value1, ASHashableValue value2)
{
	register char *str1 = (char*)value1;
	register char *str2 = (char*)value2;
	register int   i = 0;

	if (str1 == str2)
		return 0;
	if (str1 == NULL)
		return -1;
	if (str2 == NULL)
		return 1;
	do
	{
		int          u1, u2;

		u1 = str1[i];
		u2 = str2[i];
		if (islower (u1))
			u1 = toupper (u1);
		if (islower (u2))
			u2 = toupper (u2);
		if (u1 != u2)
			return (long)u1 - (long)u2;
	}while( str1[i++] );
	return 0;
}

int
asim_get_drawable_size (Drawable d, unsigned int *ret_w, unsigned int *ret_h)
{
	Display *dpy = get_default_asvisual()->dpy;
	*ret_w = 0;
	*ret_h = 0;
#ifndef X_DISPLAY_MISSING
	if( dpy && d )
	{
		Window        root;
		unsigned int  ujunk;
		int           junk;
		if (XGetGeometry (dpy, d, &root, &junk, &junk, ret_w, ret_h, &ujunk, &ujunk) != 0)
			return 1;
	}
#endif
	return 0;
}

#ifdef X_DISPLAY_MISSING
int XParseGeometry (  char *string,int *x,int *y,
                      unsigned int *width,    /* RETURN */
					  unsigned int *height)    /* RETURN */
{
	show_error( "Parsing of geometry is not supported without either Xlib opr libAfterBase" );
	return 0;
}
void XDestroyImage( void* d){}
int XGetWindowAttributes( void*d, Window w, unsigned long m, void* s){  return 0;}
void *XGetImage( void* dpy,Drawable d,int x,int y,unsigned int width,unsigned int height, unsigned long m,int t)
{return NULL ;}
unsigned long XGetPixel(void* d, int x, int y){return 0;}
int XQueryColors(void* a,Colormap c,void* x,int m){return 0;}
#endif


/***************************************/
/* from sleep.c                        */
/***************************************/
#if TIME_WITH_SYS_TIME
# include <sys/time.h>
# include <time.h>
#else
# if HAVE_SYS_TIME_H
#  include <sys/time.h>
# else
#  include <time.h>
# endif
#endif
#ifndef _WIN32
# include <sys/times.h>
#endif
static clock_t _as_ticker_last_tick = 0;
static clock_t _as_ticker_tick_size = 1;
static clock_t _as_ticker_tick_time = 0;

/**************************************************************************
 * Sleep for n microseconds
 *************************************************************************/
void
sleep_a_little (int n)
{
#ifndef _WIN32
	struct timeval value;

	if (n <= 0)
		return;

	value.tv_usec = n % 1000000;
	value.tv_sec = n / 1000000;

#ifndef PORTABLE_SELECT
#ifdef __hpux
#define PORTABLE_SELECT(w,i,o,e,t)	select((w),(int *)(i),(int *)(o),(e),(t))
#else
#define PORTABLE_SELECT(w,i,o,e,t)	select((w),(i),(o),(e),(t))
#endif
#endif
	PORTABLE_SELECT (1, 0, 0, 0, &value);
#else /* win32 : */
	Sleep(n);
#endif
}

void
asim_start_ticker (unsigned int size)
{
#ifndef _WIN32
	struct tms    t;

	_as_ticker_last_tick = times (&t);		   /* in system ticks */
	if (_as_ticker_tick_time == 0)
	{
		register clock_t delta = _as_ticker_last_tick;
		/* calibrating clock - how many ms per cpu tick ? */
		sleep_a_little (100);
		_as_ticker_last_tick = times (&t);
		delta = _as_ticker_last_tick - delta ;
		if( delta <= 0 )
			_as_ticker_tick_time = 100;
		else
			_as_ticker_tick_time = 101 / delta;
	}
#else
	_as_ticker_tick_time = 1000;
	_as_ticker_last_tick = time(NULL) ;
#endif
	_as_ticker_tick_size = size;			   /* in ms */

}

void
asim_wait_tick ()
{
#ifndef _WIN32
	struct tms    t;
	register clock_t curr = (times (&t) - _as_ticker_last_tick) * _as_ticker_tick_time;
#else
	register int curr = (time(NULL) - _as_ticker_last_tick) * _as_ticker_tick_time;
#endif

	if (curr < _as_ticker_tick_size)
		sleep_a_little (_as_ticker_tick_size - curr);

#ifndef _WIN32
	_as_ticker_last_tick = times (&t);
#else
	_as_ticker_last_tick = time(NULL) ;
#endif
}

#ifndef _WIN32
/*
 * Non-NULL select and dcomp pointers are *NOT* tested, but should be OK.
 * They are not used by afterstep however, so this implementation should
 * be good enough.
 *
 * c.ridd@isode.com
 */
int
asim_my_scandir_ext ( const char *dirname, int (*filter_func) (const char *),
				 Bool (*handle_direntry_func)( const char *fname, const char *fullname, struct stat *stat_info, void *aux_data), 
				 void *aux_data)
{
	DIR          *d;
	struct dirent *e;						   /* Pointer to static struct inside readdir() */
	int           n = 0;					   /* Count of nl used so far */
	char         *filename;					   /* For building filename to pass to stat */
	char         *p;						   /* Place where filename starts */
	struct stat   stat_info;

	d = opendir (dirname);

	if (d == NULL)
		return -1;

	filename = (char *)safecalloc (1, strlen (dirname) + PATH_MAX + 2);
	if (filename == NULL)
	{
		closedir (d);
		return -1;
	}
	strcpy (filename, dirname);
	p = filename + strlen (filename);
	if( *p != '/' )
	{	
		*p++ = '/';
		*p = 0;									   /* Just in case... */
	}
	
	while ((e = readdir (d)) != NULL)
	{
		if ((filter_func == NULL) || filter_func (&(e->d_name[0])))
		{
			int i = 0; 
			/* Fill in the fields using stat() */
			do{ p[i] = e->d_name[i]; ++i ; }while(  e->d_name[i] && i < PATH_MAX ); 
			p[i] ='\0' ;
			if (stat (filename, &stat_info) != -1)
			{	
				if( handle_direntry_func( e->d_name, filename, &stat_info, aux_data) )
					n++;
			}
		}
	}
	free (filename);

	if (closedir (d) == -1)
		return -1;
	/* Return the count of the entries */
	return n;
}

#endif /* #ifndef _WIN32 */

/***************************************/
/* from xml.c                          */
/***************************************/
static char* cdata_str = XML_CDATA_STR;
static char* container_str = XML_CONTAINER_STR;
static ASHashTable *asxml_var = NULL;

void
asim_asxml_var_init(void)
{
	if ( asxml_var == NULL )
	{
		Display *dpy = get_default_asvisual()->dpy;

    	asxml_var = create_ashash(0, string_hash_value, string_compare, string_destroy_without_data);
    	if (!asxml_var) return;
#ifndef X_DISPLAY_MISSING
    	if ( dpy != NULL )
		{
        	asxml_var_insert("xroot.width",  XDisplayWidth (dpy, DefaultScreen(dpy)));
        	asxml_var_insert("xroot.height", XDisplayHeight(dpy, DefaultScreen(dpy)));
      	}
#endif
	}
}

void
asim_asxml_var_insert(const char* name, int value)
{
	ASHashData hdata;

    if (!asxml_var) asxml_var_init();
    if (!asxml_var) return;

    /* Destroy any old data associated with this name. */
    remove_hash_item(asxml_var, AS_HASHABLE(name), NULL, True);

    show_progress("Defining var [%s] == %d.", name, value);

    hdata.i = value;
    add_hash_item(asxml_var, AS_HASHABLE(mystrdup(name)), hdata.vptr);
}

int
asim_asxml_var_get(const char* name)
{
	ASHashData hdata = {0};

    if (!asxml_var) asxml_var_init();
    if (!asxml_var) return 0;
    if( get_hash_item(asxml_var, AS_HASHABLE(name), &hdata.vptr) != ASH_Success ) 
	{	
		show_debug(__FILE__, "asxml_var_get", __LINE__, "Use of undefined variable [%s].", name);
		return 0;
	}
    return hdata.i;
}

static int
asim_asxml_var_nget(char* name, int n) {
      int value;
      char oldc = name[n];
      name[n] = '\0';
      value = asxml_var_get(name);
      name[n] = oldc;
      return value;
}

void
asim_asxml_var_cleanup(void)
{
	if ( asxml_var != NULL )
    	destroy_ashash( &asxml_var );

}

static char* lcstring(char* str) 
{
	char* ptr = str;
	for ( ; *ptr ; ptr++) if (isupper((int)*ptr)) *ptr = tolower((int)*ptr);
	return str;
}


static xml_elem_t* xml_elem_new(void) {
	xml_elem_t* elem = NEW(xml_elem_t);
	elem->next = elem->child = NULL;
	elem->parm = elem->tag = NULL;
	elem->tag_id = XML_UNKNOWN_ID ;
/*	LOCAL_DEBUG_OUT("elem = %p", elem); */
	return elem;
}

static int 
xml_name2id( const char *name, ASHashTable *vocabulary )
{
	ASHashData hdata;
	hdata.i = 0 ;
    get_hash_item(vocabulary, AS_HASHABLE(name), &hdata.vptr); 
	return hdata.i;		
}	 

static xml_elem_t* xml_elem_remove(xml_elem_t** list, xml_elem_t* elem) {
	/* Splice the element out of the list, if it's in one. */
	if (list) {
		if (*list == elem) {
			*list = elem->next;
		} else {
			xml_elem_t* ptr;
			for (ptr = *list ; ptr->next ; ptr = ptr->next) {
				if (ptr->next == elem) {
					ptr->next = elem->next;
					break;
				}
			}
		}
	}
	elem->next = NULL;
	return elem;
}

static void xml_insert(xml_elem_t* parent, xml_elem_t* child) {
	child->next = NULL;
	if (!parent->child) {
		parent->child = child;
		return;
	}
	for (parent = parent->child ; parent->next ; parent = parent->next);
	parent->next = child;
}

xml_elem_t* asim_xml_parse_parm(const char* parm, ASHashTable *vocabulary) {
	xml_elem_t* list = NULL;
	const char* eparm;

	if (!parm) return NULL;

	for (eparm = parm ; *eparm ; ) {
		xml_elem_t* p;
		const char* bname;
		const char* ename;
		const char* bval;
		const char* eval;

		/* Spin past any leading whitespace. */
		for (bname = eparm ; isspace((int)*bname) ; bname++);

		/* Check for a parm.  First is the parm name. */
		for (ename = bname ; xml_tagchar((int)*ename) ; ename++);

		/* No name equals no parm equals broken tag. */
		if (!*ename) { eparm = NULL; break; }

		/* No "=" equals broken tag.  We do not support HTML-style parms */
		/* with no value.                                                */
		for (bval = ename ; isspace((int)*bval) ; bval++);
		if (*bval != '=') { eparm = NULL; break; }

		while (isspace((int)*++bval));

		/* If the next character is a quote, spin until we see another one. */
		if (*bval == '"' || *bval == '\'') {
			char quote = *bval;
			bval++;
			for (eval = bval ; *eval && *eval != quote ; eval++);
		} else {
			for (eval = bval ; *eval && !isspace((int)*eval) ; eval++);
		}

		for (eparm = eval ; *eparm && !isspace((int)*eparm) ; eparm++);

		/* Add the parm to our list. */
		p = xml_elem_new();
		if (!list) list = p;
		else { p->next = list; list = p; }
		p->tag = lcstring(mystrndup(bname, ename - bname));
		if( vocabulary )
			p->tag_id = xml_name2id( p->tag, vocabulary );
		p->parm = mystrndup(bval, eval - bval);
	}

	if (!eparm) {
		while (list) {
			xml_elem_t* p = list->next;
			free(list->tag);
			free(list->parm);
			free(list);
			list = p;
		}
	}

	return list;
}


void asim_xml_elem_delete(xml_elem_t** list, xml_elem_t* elem) {
/*	LOCAL_DEBUG_OUT("elem = %p", elem); */

	if (list) xml_elem_remove(list, elem);
	while (elem) {
		xml_elem_t* ptr = elem;
		elem = elem->next;
		if (ptr->child) xml_elem_delete(NULL, ptr->child);
		if (ptr->tag && ptr->tag != cdata_str && ptr->tag != container_str) free(ptr->tag);
		if (ptr->parm) free(ptr->parm);
		free(ptr);
	}
}

static xml_elem_t *
create_CDATA_tag()	
{ 
	xml_elem_t *cdata = xml_elem_new();
	cdata->tag = strdup(XML_CDATA_STR) ;
	cdata->tag_id = XML_CDATA_ID ;
	return cdata;
}

static xml_elem_t *
create_CONTAINER_tag()	
{ 
	xml_elem_t *container = xml_elem_new();
	container->tag = strdup(XML_CONTAINER_STR) ;
	container->tag_id = XML_CONTAINER_ID ;
	return container;
}



xml_elem_t* asim_xml_parse_doc(const char* str, ASHashTable *vocabulary) {
	xml_elem_t* elem = create_CONTAINER_tag();
	xml_parse(str, elem, vocabulary);
	return elem;
}

int asim_xml_parse(const char* str, xml_elem_t* current, ASHashTable *vocabulary) {
	const char* ptr = str;

	/* Find a tag of the form <tag opts>, </tag>, or <tag opts/>. */
	while (*ptr) {
		const char* oab = ptr;

		/* Look for an open oab bracket. */
		for (oab = ptr ; *oab && *oab != '<' ; oab++);

		/* If there are no oab brackets left, we're done. */
		if (*oab != '<') return oab - str;

		/* Does this look like a close tag? */
		if (oab[1] == '/') 
		{
			const char* etag;
			/* Find the end of the tag. */
			for (etag = oab + 2 ; xml_tagchar((int)*etag) ; etag++);

			while (isspace((int)*etag)) ++etag;
			/* If this is an end tag, and the tag matches the tag we're parsing, */
			/* we're done.  If not, continue on blindly. */
			if (*etag == '>') 
			{
				if (!mystrncasecmp(oab + 2, current->tag, etag - (oab + 2))) 
				{
					if (oab - ptr) 
					{
						xml_elem_t* child = create_CDATA_tag();
						child->parm = mystrndup(ptr, oab - ptr);
						xml_insert(current, child);
					}
					return (etag + 1) - str;
				}
			}

			/* This tag isn't interesting after all. */
			ptr = oab + 1;
		}

		/* Does this look like a start tag? */
		if (oab[1] != '/') {
			int empty = 0;
			const char* btag = oab + 1;
			const char* etag;
			const char* bparm;
			const char* eparm;

			/* Find the end of the tag. */
			for (etag = btag ; xml_tagchar((int)*etag) ; etag++);

			/* If we reached the end of the document, continue on. */
			if (!*etag) { ptr = oab + 1; continue; }

			/* Find the beginning of the parameters, if they exist. */
			for (bparm = etag ; isspace((int)*bparm) ; bparm++);

			/* From here on, we're looking for a sequence of parms, which have
			 * the form [a-z0-9-]+=("[^"]"|'[^']'|[^ \t\n]), followed by either
			 * a ">" or a "/>". */
			for (eparm = bparm ; *eparm ; ) {
				const char* tmp;

				/* Spin past any leading whitespace. */
				for ( ; isspace((int)*eparm) ; eparm++);

				/* Are we at the end of the tag? */
				if (*eparm == '>' || (*eparm == '/' && eparm[1] == '>')) break;

				/* Check for a parm.  First is the parm name. */
				for (tmp = eparm ; xml_tagchar((int)*tmp) ; tmp++);

				/* No name equals no parm equals broken tag. */
				if (!*tmp) { eparm = NULL; break; }

				/* No "=" equals broken tag.  We do not support HTML-style parms
				   with no value. */
				for ( ; isspace((int)*tmp) ; tmp++);
				if (*tmp != '=') { eparm = NULL; break; }

				do { ++tmp; } while (isspace((int)*tmp));

				/* If the next character is a quote, spin until we see another one. */
				if (*tmp == '"' || *tmp == '\'') {
					char quote = *tmp;
					for (tmp++ ; *tmp && *tmp != quote ; tmp++);
				}

				/* Now look for a space or the end of the tag. */
				for ( ; *tmp && !isspace((int)*tmp) && *tmp != '>' && !(*tmp == '/' && tmp[1] == '>') ; tmp++);

				/* If we reach the end of the string, there cannot be a '>'. */
				if (!*tmp) { eparm = NULL; break; }

				/* End of the parm.  */
				eparm = tmp;
				
				if (!isspace((int)*tmp)) break; 
				for ( ; isspace((int)*tmp) ; tmp++);
				if( *tmp == '>' || (*tmp == '/' && tmp[1] == '>') )
					break;
			}

			/* If eparm is NULL, the parm string is invalid, and we should
			 * abort processing. */
			if (!eparm) { ptr = oab + 1; continue; }

			/* Save CDATA, if there is any. */
			if (oab - ptr) {
				xml_elem_t* child = create_CDATA_tag();
				child->parm = mystrndup(ptr, oab - ptr);
				xml_insert(current, child);
			}

			/* We found a tag!  Advance the pointer. */
			for (ptr = eparm ; isspace((int)*ptr) ; ptr++);
			empty = (*ptr == '/');
			ptr += empty + 1;

			/* Add the tag to our children and parse it. */
			{
				xml_elem_t* child = xml_elem_new();
				child->tag = lcstring(mystrndup(btag, etag - btag));
				if( vocabulary )
					child->tag_id = xml_name2id( child->tag, vocabulary );
				if (eparm - bparm) child->parm = mystrndup(bparm, eparm - bparm);
				xml_insert(current, child);
				if (!empty) ptr += xml_parse(ptr, child, vocabulary);
			}
		}
	}
	return ptr - str;
}


char *asim_interpret_ctrl_codes( char *text )
{
	register char *ptr = text ;
	int len, curr = 0 ;
	if( ptr == NULL )  return NULL ;	

	len = strlen(ptr);
	while( ptr[curr] != '\0' ) 
	{
		if( ptr[curr] == '\\' && ptr[curr+1] != '\0' ) 	
		{
			char subst = '\0' ;
			switch( ptr[curr+1] ) 
			{
				case '\\': subst = '\\' ;  break ;	
				case 'a' : subst = '\a' ;  break ;	 
				case 'b' : subst = '\b' ;  break ;	 
				case 'f' : subst = '\f' ;  break ;	 
				case 'n' : subst = '\n' ;  break ;	 
				case 'r' : subst = '\r' ;  break ;	
				case 't' : subst = '\t' ;  break ;	
				case 'v' : subst = '\v' ;  break ;	 
			}	 
			if( subst ) 
			{
				register int i = curr ; 
				ptr[i] = subst ;
				while( ++i < len ) 
					ptr[i] = ptr[i+1] ; 
				--len ; 
			}
		}	 
		++curr ;
	}	 
	return text;
}	 

void asim_reset_xml_buffer( ASXmlBuffer *xb )
{
	if( xb ) 
	{
		xb->current = xb->used = 0 ; 
		xb->state = ASXML_Start	 ;
		xb->level = 0 ;
		xb->verbatim = False ;
		xb->quoted = False ;
		xb->tag_type = ASXML_OpeningTag ;
		xb->tags_count = 0 ;
	}		  
}	 

void 
asim_free_xml_buffer_resources (ASXmlBuffer *xb)
{
	if (xb && xb->buffer)
	{
		free (xb->buffer);
		xb->allocated = xb->current = xb->used = 0 ; 
		xb->buffer = NULL;
	}
}

static inline void
realloc_xml_buffer( ASXmlBuffer *xb, int len )
{
	if( xb->used + len > xb->allocated ) 
	{	
		xb->allocated = xb->used + (((len>>11)+1)<<11) ;	  
		xb->buffer = realloc( xb->buffer, xb->allocated );
	}
}

void 
asim_add_xml_buffer_chars( ASXmlBuffer *xb, char *tmp, int len )
{
	realloc_xml_buffer (xb, len);
	memcpy( &(xb->buffer[xb->used]), tmp, len );
	xb->used += len ;
}

static void 
add_xml_buffer_spaces( ASXmlBuffer *xb, int len )
{
	if (len > 0)
	{
		realloc_xml_buffer (xb, len);
		memset( &(xb->buffer[xb->used]), ' ', len );
		xb->used += len ;
	}
}

static void 
add_xml_buffer_open_tag( ASXmlBuffer *xb, xml_elem_t *tag )
{
	int tag_len = strlen (tag->tag);
	int parm_len = 0;
	xml_elem_t* parm = NULL ; 
	
	if (tag->parm)
	{
		xml_elem_t *t = parm = xml_parse_parm(tag->parm, NULL);
		while (t)
		{
			parm_len += 1 + strlen(t->tag) + 1 + 1 + strlen(t->parm) + 1;
			t = t->next;
		}
	}
	realloc_xml_buffer (xb, 1+tag_len+1+parm_len+2);
	xb->buffer[(xb->used)++] = '<';
	memcpy (&(xb->buffer[xb->used]), tag->tag, tag_len);
	xb->used += tag_len ;

	while (parm) 
	{
		xml_elem_t* p = parm->next;
		int len;
		xb->buffer[(xb->used)++] = ' ';
		for (len = 0 ; parm->tag[len] ; ++len)
			xb->buffer[xb->used+len] = parm->tag[len];
		xb->used += len ;
		xb->buffer[(xb->used)++] = '=';
		xb->buffer[(xb->used)++] = '\"';
		for (len = 0 ; parm->parm[len] ; ++len)
			xb->buffer[xb->used+len] = parm->parm[len];
		xb->used += len ;
		xb->buffer[(xb->used)++] = '\"';
		free(parm->tag);
		free(parm->parm);
		free(parm);
		parm = p;
	}

	if (tag->child == NULL)
		xb->buffer[(xb->used)++] = '/';
	xb->buffer[(xb->used)++] = '>';
}

static void 
add_xml_buffer_close_tag( ASXmlBuffer *xb, xml_elem_t *tag )
{
	int tag_len = strlen (tag->tag);
	realloc_xml_buffer (xb, tag_len+3);
	xb->buffer[(xb->used)++] = '<';
	xb->buffer[(xb->used)++] = '/';
	memcpy (&(xb->buffer[xb->used]), tag->tag, tag_len);
	xb->used += tag_len ;
	xb->buffer[(xb->used)++] = '>';
}

int 
asim_spool_xml_tag( ASXmlBuffer *xb, char *tmp, int len )
{
	register int i = 0 ; 
	
	if( !xb->verbatim && !xb->quoted && 
		(xb->state != ASXML_Start || xb->level == 0 )) 
	{	/* skip spaces if we are not in string */
		while( i < len && isspace( (int)tmp[i] )) ++i;
		if( i >= len ) 
			return i;
	}
	if( xb->state == ASXML_Start ) 
	{     /* we are looking for the opening '<' */
		if( tmp[i] != '<' ) 
		{
			if( xb->level == 0 ) 	  
				xb->state = ASXML_BadStart ; 
			else
			{
				int start = i ; 
				while( i < len && tmp[i] != '<' ) ++i ;	  
				add_xml_buffer_chars( xb, &tmp[start], i - start );
				return i;
			}
		}else
		{	
			xb->state = ASXML_TagOpen; 	
			xb->tag_type = ASXML_OpeningTag ;
			add_xml_buffer_chars( xb, "<", 1 );
			if( ++i >= len ) 
				return i;
		}
	}
	
	if( xb->state == ASXML_TagOpen ) 
	{     /* we are looking for the beginning of tag name  or closing tag's slash */
		if( tmp[i] == '/' ) 
		{
			xb->state = ASXML_TagName; 
			xb->verbatim = True ; 		   
			xb->tag_type = ASXML_ClosingTag ;
			add_xml_buffer_chars( xb, "/", 1 );
			if( ++i >= len ) 
				return i;
		}else if( isalnum((int)tmp[i]) )	
		{	 
			xb->state = ASXML_TagName; 		   
			xb->verbatim = True ; 		   
		}else
			xb->state = ASXML_BadTagName ;
	}

	if( xb->state == ASXML_TagName ) 
	{     /* we are looking for the tag name */
		int start = i ;
		/* need to store attribute name in form : ' attr_name' */
		while( i < len && isalnum((int)tmp[i]) ) ++i ;
		if( i > start ) 
			add_xml_buffer_chars( xb, &tmp[start], i - start );
		if( i < len ) 
		{	
			if( isspace( (int)tmp[i] ) || tmp[i] == '>' ) 
			{
				xb->state = ASXML_TagAttrOrClose;
				xb->verbatim = False ; 
			}else
				xb->state = ASXML_BadTagName ;
		}			 
		return i;
	}

	if( xb->state == ASXML_TagAttrOrClose ) 
	{   /* we are looking for the atteribute or closing '/>' or '>' */
		Bool has_slash = (xb->tag_type != ASXML_OpeningTag);

		if( !has_slash && tmp[i] == '/' )
		{	
			xb->tag_type = ASXML_SimpleTag ;
			add_xml_buffer_chars( xb, "/", 1 );		 			  
			++i ;
			has_slash = True ;
		}
		if( i < len ) 
		{	
			if( has_slash && tmp[i] != '>') 
				xb->state = ASXML_UnexpectedSlash ;	  
			else if( tmp[i] == '>' ) 
			{
				++(xb->tags_count);
				xb->state = ASXML_Start; 	
	 			add_xml_buffer_chars( xb, ">", 1 );		 			  
				++i ;
				if( xb->tag_type == ASXML_OpeningTag )
					++(xb->level);
				else if( xb->tag_type == ASXML_ClosingTag )					
				{
					if( xb->level <= 0 )
					{
				 		xb->state = ASXML_UnmatchedClose;
						return i;		   
					}else
						--(xb->level);			
				}		 			   
			}else if( !isalnum( (int)tmp[i] ) )	  
				xb->state = ASXML_BadAttrName ;
			else
			{	
				xb->state = ASXML_AttrName;		 
				xb->verbatim = True ;
				add_xml_buffer_chars( xb, " ", 1);
			}
		}
		return i;
	}

	if( xb->state == ASXML_AttrName ) 
	{	
		int start = i ;
		/* need to store attribute name in form : ' attr_name' */
		while( i < len && isalnum((int)tmp[i]) ) ++i ;
		if( i > start ) 
			add_xml_buffer_chars( xb, &tmp[start], i - start );
		if( i < len ) 
		{	
			if( isspace( (int)tmp[i] ) || tmp[i] == '=' ) 
			{
				xb->state = ASXML_AttrEq;
				xb->verbatim = False ; 
				/* should fall down to case below */
			}else
				xb->state = ASXML_BadAttrName ;
		}
	 	return i;				 
	}	

	if( xb->state == ASXML_AttrEq )                   /* looking for '=' */
	{
		if( tmp[i] == '=' ) 
		{
			xb->state = ASXML_AttrValueStart;				
			add_xml_buffer_chars( xb, "=", 1 );		 			  
			++i ;
		}else	 
			xb->state = ASXML_MissingAttrEq ;
		return i;
	}	
	
	if( xb->state == ASXML_AttrValueStart )/*looking for attribute value:*/
	{
		xb->state = ASXML_AttrValue ;
		if( tmp[i] == '"' )
		{
			xb->quoted = True ; 
			add_xml_buffer_chars( xb, "\"", 1 );
			++i ;
		}else	 
			xb->verbatim = True ; 
		return i;
	}	  
	
	if( xb->state == ASXML_AttrValue )  /* looking for attribute value : */
	{
		if( !xb->quoted && isspace((int)tmp[i]) ) 
		{
			add_xml_buffer_chars( xb, " ", 1 );
			++i ;
			xb->verbatim = False ; 
			xb->state = ASXML_TagAttrOrClose ;
		}else if( xb->quoted && tmp[i] == '"' ) 
		{
			add_xml_buffer_chars( xb, "\"", 1 );
			++i ;
			xb->quoted = False ; 
			xb->state = ASXML_TagAttrOrClose ;
		}else if( tmp[i] == '/' && !xb->quoted)
		{
			xb->state = ASXML_AttrSlash ;				
			add_xml_buffer_chars( xb, "/", 1 );		 			  
			++i ;
		}else if( tmp[i] == '>' )
		{
			xb->quoted = False ; 
			xb->verbatim = False ; 
			xb->state = ASXML_TagAttrOrClose ;				
		}else			
		{
			add_xml_buffer_chars( xb, &tmp[i], 1 );
			++i ;
		}
		return i;
	}	  
	if( xb->state == ASXML_AttrSlash )  /* looking for attribute value : */
	{
		if( tmp[i] == '>' )
		{
			xb->tag_type = ASXML_SimpleTag ;
			add_xml_buffer_chars( xb, ">", 1 );		 			  
			++i ;
			++(xb->tags_count);
			xb->state = ASXML_Start; 	
			xb->quoted = False ; 
			xb->verbatim = False ; 
		}else
		{
			xb->state = ASXML_AttrValue ;
		}		 
		return i;
	}

	return (i==0)?1:i;
}	   

/* reverse transformation - put xml tags into a buffer */
Bool 
asim_xml_tags2xml_buffer( xml_elem_t *tags, ASXmlBuffer *xb, int tags_count, int depth)
{
	Bool new_line = False; 

	while (tags && tags_count != 0) /* not a bug - negative tags_count means unlimited !*/
	{
		if (tags->tag_id == XML_CDATA_ID || !strcmp(tags->tag, cdata_str)) 
		{
			/* TODO : add handling for cdata with quotes, amps and gt, lt */
			add_xml_buffer_chars( xb, tags->parm, strlen(tags->parm));
		}else 
		{
			if (depth >= 0 && (tags->child != NULL || tags->next != NULL)) 
			{
				add_xml_buffer_chars( xb, "\n", 1);
				add_xml_buffer_spaces( xb, depth*2);
				new_line = True ;	  
			}
			add_xml_buffer_open_tag( xb, tags);

			if (tags->child) 
			{
				if( xml_tags2xml_buffer( tags->child, xb, -1, (depth < 0)?-1:depth+1 ))
				{
					if (depth >= 0)
					{
						add_xml_buffer_chars( xb, "\n", 1);
						add_xml_buffer_spaces( xb, depth*2);
					}
				}
				add_xml_buffer_close_tag( xb, tags);
			}
		}		
		tags = tags->next;
		--tags_count;
	}
	return new_line;
}

void asim_xml_print(xml_elem_t* root) 
{
	ASXmlBuffer xb;
	memset( &xb, 0x00, sizeof(xb));
	xml_tags2xml_buffer( root, &xb, -1, 0);
	add_xml_buffer_chars( &xb, "\0", 1 );
	printf ("%s", xb.buffer);
	free_xml_buffer_resources (&xb);
}


xml_elem_t *
asim_format_xml_buffer_state (ASXmlBuffer *xb)
{
	xml_elem_t *state_xml = NULL; 
	if (xb->state < 0) 
	{
		state_xml = xml_elem_new();
		state_xml->tag = strdup("error");
		state_xml->parm = safemalloc (64);
		sprintf(state_xml->parm, "code=%d level=%d tag_count=%d", xb->state, xb->level ,xb->tags_count );
		state_xml->child = create_CDATA_tag();
		switch( xb->state ) 
		{
			case ASXML_BadStart : state_xml->child->parm = strdup("Text encountered before opening tag bracket - not XML format"); break;
			case ASXML_BadTagName : state_xml->child->parm = strdup("Invalid characters in tag name" );break;
			case ASXML_UnexpectedSlash : state_xml->child->parm = strdup("Unexpected '/' encountered");break;
			case ASXML_UnmatchedClose : state_xml->child->parm = strdup("Closing tag encountered without opening tag" );break;
			case ASXML_BadAttrName : state_xml->child->parm = strdup("Invalid characters in attribute name" );break;
			case ASXML_MissingAttrEq : state_xml->child->parm = strdup("Attribute name not followed by '=' character" );break;
			default:
				state_xml->child->parm = strdup("Premature end of the input");break;
		}
	}else if (xb->state == ASXML_Start)
	{
		if (xb->tags_count > 0)
		{
			state_xml = xml_elem_new();
			state_xml->tag = strdup("success");
			state_xml->parm = safemalloc(64);
			sprintf(state_xml->parm, "tag_count=%d level=%d", xb->tags_count, xb->level );
		}
	}else
	{
		/* TODO */
	}
	return state_xml;
}
