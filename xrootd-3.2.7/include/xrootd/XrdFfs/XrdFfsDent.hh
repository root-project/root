/******************************************************************************/
/* XrdFfsDent.hh  help functions to merge direntries                          */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

#ifdef __cplusplus
  extern "C" {
#endif

struct XrdFfsDentnames {
    char *name;
    struct XrdFfsDentnames *next;
}; 

void XrdFfsDent_names_del(struct XrdFfsDentnames **p);
void XrdFfsDent_names_add(struct XrdFfsDentnames **p, char *name);
void XrdFfsDent_names_join(struct XrdFfsDentnames **p, struct XrdFfsDentnames **n);
int  XrdFfsDent_names_extract(struct XrdFfsDentnames **p, char ***dnarray);

void XrdFfsDent_cache_init();
void XrdFfsDent_cache_destroy();
int  XrdFfsDent_cache_fill(char *dname, char ***dnarray, int nents);
int  XrdFfsDent_cache_search(char *dname, char *dentname);

#ifdef __cplusplus
  }
#endif


