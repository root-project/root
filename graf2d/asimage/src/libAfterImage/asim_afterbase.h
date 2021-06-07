#ifndef ASIM_AFTERBASE_H_HEADER_INCLUDED
#define ASIM_AFTERBASE_H_HEADER_INCLUDED

#ifdef HAVE_MALLOC_H
# include <malloc.h>
#endif
#ifdef HAVE_STDLIB_H
# include <stdlib.h>
#endif
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
   

/* our own version of X Wrapper : */
#include "xwrap.h"

/* the goal of this header is to provide sufficient code so that
   libAfterImage could live without libAfterBase at all.
   Basically with define macros and copy over few functions
   from libAfterBase
 */

/* from libAfterBase/astypes.h : */

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


#ifdef __GNUC__

#ifndef MIN
#define MIN(x,y)                                \
  ({ const typeof(x) _x = (x); const typeof(y) _y = (y); \
     (void) (&_x == &_y);                       \
     _x < _y ? _x : _y; })
#endif
#ifndef MAX
#define MAX(x,y)                                \
  ({ const typeof(x) _x = (x); const typeof(y) _y = (y); \
     (void) (&_x == &_y);                       \
     _x > _y ? _x : _y; })
#endif

#define AS_ASSERT(p)            ((p)==(typeof(p))0)
#define AS_ASSERT_NOTVAL(p,v)   ((p)!=(typeof(p))v)

#else

#define MIN(a,b)            ((a)<(b) ? (a) : (b))
#define MAX(a,b)            ((a)>(b) ? (a) : (b))
#define AS_ASSERT(p)            ((p)==0)
#define AS_ASSERT_NOTVAL(p,v)   ((p)!=(v))
#define inline

#endif

#ifdef __INTEL_COMPILER
#define inline
#endif

#ifndef max
#define max(x,y)            MAX(x,y)
#endif

#ifndef min
#define min(x,y)            MIN(x,y)
#endif

typedef unsigned long ASFlagType ;
#define ASFLAGS_EVERYTHING  0xFFFFFFFF
typedef ASFlagType ASFlagsXref[5];

#define get_flags(var, val) 	((var) & (val))  /* making it sign safe */
#define set_flags(var, val) 	((var) |= (val))
#define clear_flags(var, val) 	((var) &= ~(val))
#define CheckSetFlag(b,f,v) 	{if((b)) (f) |= (v) ; else (f) &= ~(v);}

#define PTR2CARD32(p)       ((CARD32)(p))
#define LONG2CARD32(l)      ((CARD32)(l))

typedef struct ASMagic
{ /* just so we can safely cast void* to query magic number :*/
    unsigned long magic ;
}ASMagic;

/* from libAfterBase/selfdiag.h : */
#define get_caller_func() "unknown"

/* from libAfterBase/output.h : */
/* user app must export these if libAfterBase is not used : */
void asim_set_application_name (char *argv0);
const char *asim_get_application_name();

#define set_application_name asim_set_application_name
#define get_application_name asim_get_application_name

/*
 * FEW PRESET LEVELS OF OUTPUT :
 */
#define OUTPUT_LEVEL_INVALID        0
#define OUTPUT_LEVEL_PARSE_ERR      1
#define OUTPUT_LEVEL_ERROR          1
#define OUTPUT_LEVEL_WARNING        4
#define OUTPUT_DEFAULT_THRESHOLD    5
#define OUTPUT_LEVEL_PROGRESS       OUTPUT_DEFAULT_THRESHOLD
#define OUTPUT_LEVEL_ACTIVITY       OUTPUT_DEFAULT_THRESHOLD
#define OUTPUT_VERBOSE_THRESHOLD    6
#define OUTPUT_LEVEL_DEBUG          10   /* anything above it is hardcore debugging */

/* AfterStep specific error and Warning handlers : */
/* Returns True if something was actually printed  */
unsigned int asim_get_output_threshold();
unsigned int asim_set_output_threshold( unsigned int threshold );
#define get_output_threshold asim_get_output_threshold
#define set_output_threshold asim_set_output_threshold

Bool asim_show_error( const char *error_format, ...);
Bool asim_show_warning( const char *warning_format, ...);
Bool asim_show_progress( const char *msg_format, ...);
Bool asim_show_debug( const char *file, const char *func, int line, const char *msg_format, ...);

#define show_error asim_show_error
#define show_warning asim_show_warning
#define show_progress asim_show_progress
#define show_debug asim_show_debug

void asim_nonGNUC_debugout( const char *format, ...);
void asim_nonGNUC_debugout_stub( const char *format, ...);
/* may be used below in case compilation problems occur.
 * Please submit a bug report if usage of any of the following generates errors on
 * your compiler . Thanks!!! */

/* Some usefull debugging macros : */
#ifdef __GNUC__

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG)||defined(DEBUG_ALL))
#define DEBUG_OUT(format,args...) \
    do{ fprintf( stderr, "%s:%s:%s:%d:>" format "\n", get_application_name(), __FILE__, __FUNCTION__, __LINE__, ## args );}while(0)
#else
#define DEBUG_OUT(format,args...)
#endif /* DEBUG */

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG_ALL))
#define LOCAL_DEBUG_OUT(format,args...) \
    do{ fprintf( stderr, "%s:%s:%s:%d:>" format "\n", get_application_name(), __FILE__, __FUNCTION__, __LINE__, ## args );}while(0)
#define LOCAL_DEBUG_CALLER_OUT(format,args...) \
    do{ fprintf( stderr, "%s:%s:%s:> called from [%s] with args(" format ")\n", get_application_name(), __FILE__, __FUNCTION__, get_caller_func(), ## args );}while(0)
#else
#define LOCAL_DEBUG_OUT(format,args...)
#define LOCAL_DEBUG_CALLER_OUT(format,args...)
#endif /* LOCAL_DEBUG */

#elif  __STDC_VERSION__ >= 199901              /* C99 standard provides support for this as well : */

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG)||defined(DEBUG_ALL))
#define DEBUG_OUT(...) \
    do{ fprintf( stderr, "%s:%s:%s:%d:>", get_application_name(), __FILE__, __FUNCTION__, __LINE__ ); \
        fprintf( stderr, __VA_ARGS__); \
        fprintf( stderr, "\n"); \
    }while(0)
#else
#define DEBUG_OUT(...)
#endif /* DEBUG */

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG_ALL))
#define LOCAL_DEBUG_OUT(...) \
    do{ fprintf( stderr, "%s:%s:%s:%d:>", get_application_name(), __FILE__, __FUNCTION__, __LINE__ ); \
        fprintf( stderr, __VA_ARGS__); \
        fprintf( stderr, "\n"); \
    }while(0)
#define LOCAL_DEBUG_CALLER_OUT(...) \
    do{ fprintf( stderr, "%s:%s:%s:> called from [%s] with args(", get_application_name(), __FILE__, get_caller_func() ); \
        fprintf( stderr, __VA_ARGS__); \
        fprintf( stderr, ")\n"); \
    }while(0)
#else
#define LOCAL_DEBUG_OUT(...)
#define LOCAL_DEBUG_CALLER_OUT(...)
#endif /* LOCAL_DEBUG */

#else  /* non __GNUC__ or C99 compliant compiler : */

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG)||defined(DEBUG_ALL))
#define DEBUG_OUT           asim_nonGNUC_debugout
#else
#define DEBUG_OUT           asim_nonGNUC_debugout_stub
#endif /* DEBUG */

#if (!defined(NO_DEBUG_OUTPUT))&&(defined(LOCAL_DEBUG)||defined(DEBUG_ALL))
#define LOCAL_DEBUG_OUT     asim_nonGNUC_debugout
#define LOCAL_DEBUG_CALLER_OUT     asim_nonGNUC_debugout_stub
#else
#define LOCAL_DEBUG_OUT            asim_nonGNUC_debugout_stub
#define LOCAL_DEBUG_CALLER_OUT     asim_nonGNUC_debugout_stub
#endif /* LOCAL_DEBUG */

#endif

#if defined(DO_CLOCKING) && !defined(NO_DEBUG_OUTPUT)
#define START_TIME(started)  time_t started = clock()
#define SHOW_TIME(s,started) fprintf (stderr, "%s " s " time (clocks): %lu mlsec\n", __FUNCTION__, ((clock () - (started))*100)/CLOCKS_PER_SEC)
#else
#define START_TIME(started)  unsigned long started = 0
#define SHOW_TIME(s,started) started = 0
#endif

/* from libAfterBase/safemalloc.h : */
#define safemalloc(s) 	malloc(s)
#define safecalloc(c,s) calloc(c,s)
#define safefree(m)   	free(m)
#define	NEW(a)              	((a *)malloc(sizeof(a)))
#define	NEW_ARRAY_ZEROED(a,b)   ((a *)calloc((b), sizeof(a)))
#define	NEW_ARRAY(a,b)     		((a *)malloc((b)*sizeof(a)))

/* from libAfterBase/mystring.h : */

char *asim_mystrdup (const char *str);
char *asim_mystrndup(const char *str, size_t n);
int asim_mystrcasecmp (const char *s1, const char *s2);
int asim_mystrncasecmp (const char *s1, const char *s2, size_t n);
#define mystrdup(s)    	 asim_mystrdup(s)
#define mystrndup(s,n)    	 asim_mystrndup(s,n)
#define mystrncasecmp(s,s2,n)    asim_mystrncasecmp(s,s2,n)
#define mystrcasecmp(s,s2)       asim_mystrcasecmp(s,s2)

/* from libAfterBase/fs.h : */
#if !defined(S_IFREG) || !defined(S_IFDIR)
# include <sys/stat.h>
#endif

#ifndef _WIN32
struct direntry
  {
    mode_t d_mode;		/* S_IFDIR if a directory */
    time_t d_mtime;
    char d_name[1];
  };
#endif
int		asim_check_file_mode (const char *file, int mode);
#define CheckFile(f) 	asim_check_file_mode(f,S_IFREG)
#define CheckDir(d) 	asim_check_file_mode(d,S_IFDIR)
char   *asim_put_file_home (const char *path_with_home);
#define put_file_home(p) asim_put_file_home(p)
char   *asim_load_file     (const char *realfilename);
#define load_file(r)     asim_load_file(r)
char   *asim_load_binary_file(const char* realfilename, long *file_size_return);
#define load_binary_file(r,s)     asim_load_binary_file(r,s)
#ifndef _WIN32
int asim_my_scandir_ext ( const char *dirname, int (*filter_func) (const char *),
				 Bool (*handle_direntry_func)( const char *fname, const char *fullname, struct stat *stat_info, void *aux_data), 
				 void *aux_data);
#define my_scandir_ext(d,f,h,a) asim_my_scandir_ext((d),(f),(h),(a))   
#endif

void unix_path2dos_path( char *path );
char   *asim_find_file (const char *file, const char *pathlist, int type);
#define find_file(f,p,t) asim_find_file(f,p,t)
char   *asim_copy_replace_envvar (char *path);
#define copy_replace_envvar(p) asim_copy_replace_envvar(p)

const char *asim_parse_argb_color( const char *color, CARD32 *pargb );
#define parse_argb_color(c,p) asim_parse_argb_color((c),(p))

double asim_parse_math(const char* str, char** endptr, double size);
#define parse_math(s,e,sz)    asim_parse_math((s),(e),(sz))

#ifdef __hpux
#define PORTABLE_SELECT(w,i,o,e,t)	select((w),(int *)(i),(int *)(o),(e),(t))
#else
#define PORTABLE_SELECT(w,i,o,e,t)	select((w),(i),(o),(e),(t))
#endif

/* from libAfterBase/socket.h : */
#ifdef WORDS_BIGENDIAN
#define as_ntohl(ui32)		(ui32)
#define as_hlton(ui32)		(ui32)
#define as_ntohl16(ui16)		(ui16)
#define as_hlton16(ui16)		(ui16)
#else
#define as_ntohl(ui32)		((((ui32)&0x000000FF)<<24)|(((ui32)&0x0000FF00)<<8)|(((ui32)&0x00FF0000)>>8)|(((ui32)&0xFF000000)>>24))
#define as_hlton(ui32)		as_ntohl(ui32)     /* conversion is symmetrical */
#define as_ntohl16(ui16)		((((ui16)&0x00FF)<<8)|(((ui16)&0xFF00)>>8))
#define as_hlton16(ui16)		as_ntohl(ui16)     /* conversion is symmetrical */
#endif

#if 0
typedef union ASHashableValue
{
  unsigned long 	   long_val;
  char 				  *string_val;
  void 				  *ptr ;
}
ASHashableValue;
#else
typedef uintptr_t ASHashableValue;
#endif

typedef union ASHashData
{
 	void  *vptr ;
 	int   *iptr ;
 	unsigned int   *uiptr ;
 	long  *lptr ;
 	unsigned long   *ulptr ;
	char  *cptr ;
	int    i ;
	unsigned int ui ;
	long   l ;
	unsigned long ul ;
	CARD32 c32 ;
	CARD16 c16 ;
	CARD8  c8 ;
}ASHashData;

#define AS_HASHABLE(v)  ((ASHashableValue)((uintptr_t)(v)))

typedef struct ASHashItem
{
  struct ASHashItem *next;
  ASHashableValue value;
  void *data;			/* optional data structure related to this
				   hashable value */
}
ASHashItem;

typedef unsigned short ASHashKey;
typedef ASHashItem *ASHashBucket;

typedef struct ASHashTable
{
  ASHashKey size;
  ASHashBucket *buckets;
  ASHashKey buckets_used;
  unsigned long items_num;

  ASHashItem *most_recent ;

    ASHashKey (*hash_func) (ASHashableValue value, ASHashKey hash_size);
  long (*compare_func) (ASHashableValue value1, ASHashableValue value2);
  void (*item_destroy_func) (ASHashableValue value, void *data);
}
ASHashTable;

typedef enum
{

  ASH_BadParameter = -3,
  ASH_ItemNotExists = -2,
  ASH_ItemExistsDiffer = -1,
  ASH_ItemExistsSame = 0,
  ASH_Success = 1
}
ASHashResult;

void 		 asim_init_ashash (ASHashTable * hash, Bool freeresources);
ASHashTable *asim_create_ashash (ASHashKey size,
			  	 ASHashKey (*hash_func) (ASHashableValue, ASHashKey),
			   	long (*compare_func) (ASHashableValue, ASHashableValue),
			   	void (*item_destroy_func) (ASHashableValue, void *));
void         asim_destroy_ashash (ASHashTable ** hash);
ASHashResult asim_add_hash_item (ASHashTable * hash, ASHashableValue value, void *data);
ASHashResult asim_get_hash_item (ASHashTable * hash, ASHashableValue value, void **trg);
ASHashResult asim_remove_hash_item (ASHashTable * hash, ASHashableValue value, void **trg, Bool destroy);

void 		 asim_flush_ashash_memory_pool();
ASHashKey 	 asim_string_hash_value (ASHashableValue value, ASHashKey hash_size);
long 		 asim_string_compare (ASHashableValue value1, ASHashableValue value2);
void		 asim_string_destroy_without_data (ASHashableValue value, void *data);
/* variation for case-unsensitive strings */
ASHashKey 	 asim_casestring_hash_value (ASHashableValue value, ASHashKey hash_size);
long 		 asim_casestring_compare (ASHashableValue value1, ASHashableValue value2);

ASHashKey asim_pointer_hash_value (ASHashableValue value, ASHashKey hash_size);

#define init_ashash(h,f) 			 asim_init_ashash(h,f)
#define create_ashash(s,h,c,d) 		 asim_create_ashash(s,h,c,d)
#define	destroy_ashash(h) 		 	 asim_destroy_ashash(h)
#define	add_hash_item(h,v,d) 		 asim_add_hash_item(h,v,d)
#define	get_hash_item(h,v,t) 		 asim_get_hash_item(h,v,t)
#define	remove_hash_item(h,v,t,d)	 asim_remove_hash_item(h,v,t,d)
#define	flush_ashash_memory_pool	 asim_flush_ashash_memory_pool

#define	string_hash_value 	 	 asim_string_hash_value
#define	pointer_hash_value 	 	 asim_pointer_hash_value
#define	string_compare 		 	 asim_string_compare
#define	string_destroy_without_data  asim_string_destroy_without_data
#define	casestring_hash_value		 asim_casestring_hash_value
#define	casestring_compare     		 asim_casestring_compare

/* from sleep.c */
void asim_start_ticker (unsigned int size);
void asim_wait_tick ();
#define start_ticker 	asim_start_ticker
#define wait_tick 		asim_wait_tick

/* TODO : add xml stuff */
/* from xml.c  */

#define xml_tagchar(a) (isalnum(a) || (a) == '-' || (a) == '_')

#define XML_CDATA_STR 		"CDATA"
#define XML_CONTAINER_STR	"CONTAINER"
#define XML_CDATA_ID		-2
#define XML_CONTAINER_ID	-1
#define XML_UNKNOWN_ID		 0

#define IsCDATA(pe)    		((pe) && (pe)->tag_id == XML_CDATA_ID)

typedef struct xml_elem_t {
	struct xml_elem_t* next;
	struct xml_elem_t* child;
	char* tag;
	int tag_id;
	char* parm;
} xml_elem_t;

typedef enum
{
	ASXML_Start 			= 0,			               
	ASXML_TagOpen 			= 1,
	ASXML_TagName 			= 2,
	ASXML_TagAttrOrClose 	= 3,
	ASXML_AttrName 			= 4,
	ASXML_AttrEq 			= 5,
	ASXML_AttrValueStart 	= 6,
	ASXML_AttrValue 		= 7,
	ASXML_AttrSlash 		= 8
} ASXML_ParserState;

typedef enum
{
	ASXML_BadStart = -1,
	ASXML_BadTagName = -2,
	ASXML_UnexpectedSlash = -3,
	ASXML_UnmatchedClose = -4,
	ASXML_BadAttrName = -5,
	ASXML_MissingAttrEq = -6
} ASXML_ParserError;

typedef struct ASXmlBuffer
{
	char *buffer ;
	int allocated, used, current ;

	int state ; 
	int level ;
	Bool verbatim;
	Bool quoted;
	
	enum
	{
		ASXML_OpeningTag = 0,
		ASXML_SimpleTag,
		ASXML_ClosingTag
	}tag_type ;

	int tags_count ;
}ASXmlBuffer;

void asim_asxml_var_insert(const char* name, int value);
int asim_asxml_var_get(const char* name);
void asim_asxml_var_init(void);
void asim_asxml_var_cleanup(void);

xml_elem_t* asim_xml_parse_parm(const char* parm, struct ASHashTable *vocabulary);

void asim_xml_elem_delete(xml_elem_t** list, xml_elem_t* elem);
xml_elem_t* asim_xml_parse_doc(const char* str, struct ASHashTable *vocabulary);
int asim_xml_parse(const char* str, xml_elem_t* current, struct ASHashTable *vocabulary);

void asim_reset_xml_buffer( ASXmlBuffer *xb );
void asim_free_xml_buffer_resources (ASXmlBuffer *xb);

void asim_add_xml_buffer_chars( ASXmlBuffer *xb, char *tmp, int len );
int asim_spool_xml_tag( ASXmlBuffer *xb, char *tmp, int len );

Bool asim_xml_tags2xml_buffer (xml_elem_t *tags, ASXmlBuffer *xb, int tags_count, int depth);
void asim_xml_print (xml_elem_t* root);
xml_elem_t *asim_format_xml_buffer_state (ASXmlBuffer *xb);

char *asim_interpret_ctrl_codes( char *text );

#define asxml_var_insert(n,v)				asim_asxml_var_insert((n),(v))
#define asxml_var_get(n)					asim_asxml_var_get((n))
#define asxml_var_init						asim_asxml_var_init
#define asxml_var_cleanup					asim_asxml_var_cleanup

#define xml_parse_parm(p,v)					asim_xml_parse_parm((p),(v))

#define xml_elem_delete(l,e)				asim_xml_elem_delete((l),(e))
#define xml_parse_doc(s,v)					asim_xml_parse_doc((s),(v))
#define xml_parse(s,c,v)					asim_xml_parse((s),(c),(v))

#define reset_xml_buffer(xb)				asim_reset_xml_buffer((xb))
#define free_xml_buffer_resources(xb)		asim_free_xml_buffer_resources((xb))

#define add_xml_buffer_chars(xb,t,l)		asim_add_xml_buffer_chars((xb),(t),(l))
#define spool_xml_tag(xb,t,l)				asim_spool_xml_tag((xb),(t),(l))

#define xml_tags2xml_buffer(t,xb,tc,d)		asim_xml_tags2xml_buffer((t),(xb),(tc),(d))
#define xml_print(r) 						asim_xml_print((r))
#define format_xml_buffer_state(xb)			asim_format_xml_buffer_state((xb))

#define interpret_ctrl_codes(t) 			asim_interpret_ctrl_codes((t))

#ifdef __cplusplus
}
#endif


#endif /* ASIM_AFTERBASE_H_HEADER_INCLUDED */

