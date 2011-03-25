/******************************************************************************/
/* XrdFfsDent.cc  help functions to merge direntries                          */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/* Author: Wei Yang (SLAC National Accelerator Laboratory, 2009)              */
/*         Contract DE-AC02-76-SFO0515 with the Department of Energy          */
/******************************************************************************/

#include "XrdFfs/XrdFfsDent.hh"

#ifdef __cplusplus
  extern "C" {
#endif

/*
  to be used by quick sort
 */
int XrdFfsDent_cstr_cmp(const void *a, const void *b)
{
    const char **aa = (const char **)a;
    const char **bb = (const char **)b;
    return strcmp(*aa, *bb);
}

/*
  _del() frees the head node of *p
 */
void XrdFfsDent_names_del(struct XrdFfsDentnames **p)
{
    (*p)->name = NULL;
    (*p)->next = NULL;
    free(*p);
} 

void XrdFfsDent_names_add(struct XrdFfsDentnames **p, char *name)
{
    struct XrdFfsDentnames *n = (struct XrdFfsDentnames*)malloc(sizeof(struct XrdFfsDentnames));
    n->name = strdup(name);

    n->next = *p;
    *p = n;
    return; 
}

/*
  _join() joins *n to *p. Note *p or *n may equal to NULL. In that
  case, old *p is remove and *p will equal to *n in the end.

  Giving list A, B, C and one want to join in the order of A->B->C 
  (so that A points to the joined final list), there are two ways 
  to to this:

  1. _join(&A, &B); _join(&A, &C) (not efficient)
  2. _join(&B, &C); _join(&A, &B) (efficient)

 */
void XrdFfsDent_names_join(struct XrdFfsDentnames **p, struct XrdFfsDentnames **n)
{
    struct XrdFfsDentnames *t, *l;

    if ( *p != NULL )
    {
        t = *p;
        while ( t != NULL ) 
        {
            l = t;
            t = t->next;
        }
        l->next = *n;
    }
    else
        *p = *n;
}

/* 
   _extract() returns (to dnarray) a char array with all (*p)->name 
   sorted accroding to strcmp(), and destroy *p.
*/
int XrdFfsDent_names_extract(struct XrdFfsDentnames **p, char ***dnarray)
{
    struct XrdFfsDentnames *x, *y;
    int i = 0;

    y = *p; 
    while (y != NULL) 
    {
        i++;
        y = y->next;
    }
    /* be careful, old dnarray is lost */
    *dnarray = (char**) malloc(sizeof(char*) * i);

    x = *p;
    y = *p; 
    i = 0;
    while (y != NULL) 
    {
        (*dnarray)[i++] = y->name;
        y = y->next;
        XrdFfsDent_names_del(&x);
        x = y;
    }

    qsort((*dnarray), i, sizeof(char*), XrdFfsDent_cstr_cmp);
    *p = NULL;
    return i;
}

/* managing caches for dentnames */

struct XrdFfsDentcache {
    time_t t0;
    time_t life;
    unsigned int nents;
    char *dirname;
    char **dnarray;
};

void XrdFfsDent_dentcache_fill(struct XrdFfsDentcache *cache, char *dname, char ***dnarray, int nents)
{
    int i;

    cache->dirname = strdup(dname);
    cache->nents = nents;
    cache->t0 = time(NULL);
    cache->life = nents / 10 ;
    cache->dnarray = (char**) malloc(sizeof(char*) * nents);
    
    for (i = 0; i < nents; i++)
        cache->dnarray[i] = strdup((*dnarray)[i]);
}

void XrdFfsDent_dentcache_free(struct XrdFfsDentcache *cache)
{
    int i;
    for (i = 0; i < (int)cache->nents; i++)
    {
        free(cache->dnarray[i]);
    }    
    cache->nents = 0;
    free(cache->dnarray);
    free(cache->dirname);
    cache->dnarray = NULL;
    cache->dirname = NULL;
}

/* expired cache may still be useful. invalid cache should not be used */
int  XrdFfsDent_dentcache_expired(struct XrdFfsDentcache *cache)
{
    time_t t1;
    t1 = time(NULL);
    return (((t1 - cache->t0) < cache->life)? 0 : 1);
}

int  XrdFfsDent_dentcache_invalid(struct XrdFfsDentcache *cache)
{
    time_t t1;
    t1 = time(NULL);
    return (((t1 - cache->t0) < 28700)? 0 : 1); // after 8 hours (28800 sec), the redirector no longer remembers
}

int  XrdFfsDent_dentcache_search(struct XrdFfsDentcache *cache, char *dname, char *dentname)
{
    char path[1024]; 

    strcpy(path, dname);
    if (dentname != NULL && path[strlen(path) -1] != '/') 
        strcat(path,"/");
    if (dentname != NULL) strcat(path, dentname);
    if (XrdFfsDent_dentcache_invalid(cache))
        return 0;
    else if (strlen(cache->dirname) == strlen(path) && strcmp(cache->dirname, path) == 0) 
        return 1;
    else if (strlen(cache->dirname) != strlen(dname) || strcmp(cache->dirname, dname) != 0) 
        return 0;
    else if (bsearch(&dentname, cache->dnarray, cache->nents, sizeof(char*), XrdFfsDent_cstr_cmp) != NULL)
        return 1;
    else
        return 0;
}

#define XrdFfsDent_NDENTCACHES 20 
struct XrdFfsDentcache XrdFfsDentCaches[XrdFfsDent_NDENTCACHES];
pthread_mutex_t XrdFfsDentCaches_mutex = PTHREAD_MUTEX_INITIALIZER;

void XrdFfsDent_cache_init()
{
    int i;
    for (i = 0; i < XrdFfsDent_NDENTCACHES; i++)
    {
        XrdFfsDentCaches[i].t0 = 0;
        XrdFfsDentCaches[i].nents = 0;
        XrdFfsDentCaches[i].dirname = strdup("");
        XrdFfsDentCaches[i].dnarray = NULL;
    }
}

int XrdFfsDent_cache_fill(char *dname, char ***dnarray, int nents)
{
    int i;
    pthread_mutex_lock(&XrdFfsDentCaches_mutex);
    for (i = 0; i < XrdFfsDent_NDENTCACHES; i++)
    {
        if (XrdFfsDent_dentcache_search(&XrdFfsDentCaches[i], dname, NULL) != 0)
        {
            XrdFfsDent_dentcache_free(&XrdFfsDentCaches[i]);
            XrdFfsDent_dentcache_fill(&XrdFfsDentCaches[i], dname, dnarray, nents); 
            pthread_mutex_unlock(&XrdFfsDentCaches_mutex);
            return 1;
        } 
    }
    for (i = 0; i < XrdFfsDent_NDENTCACHES; i++)
    {
        if (XrdFfsDent_dentcache_expired(&XrdFfsDentCaches[i]) || XrdFfsDent_dentcache_invalid(&XrdFfsDentCaches[i])) 
        {
            XrdFfsDent_dentcache_free(&XrdFfsDentCaches[i]);
            XrdFfsDent_dentcache_fill(&XrdFfsDentCaches[i], dname, dnarray, nents); 
            pthread_mutex_unlock(&XrdFfsDentCaches_mutex);
            return 1;
        }
    }
    pthread_mutex_unlock(&XrdFfsDentCaches_mutex);
    return 0;
}

int XrdFfsDent_cache_search(char *dname, char *dentname)
{
    int i, rval = 0;
    pthread_mutex_lock(&XrdFfsDentCaches_mutex);
    for (i = 0; i < XrdFfsDent_NDENTCACHES; i++)
        if (XrdFfsDent_dentcache_search(&XrdFfsDentCaches[i], dname, dentname) == 1) 
        {
            rval = 1;
            break;
        }
    pthread_mutex_unlock(&XrdFfsDentCaches_mutex);
    return rval;
}

void XrdFfsDent_cache_destroy()
{
    int i;
    for (i = 0; i < XrdFfsDent_NDENTCACHES; i++)
        XrdFfsDent_dentcache_free(&XrdFfsDentCaches[i]);
}

/*
#include <stdio.h>

main() 
{
    struct XrdFfsDentnames *x = NULL;
    struct XrdFfsDentnames *y = NULL;
    struct XrdFfsDentnames *z = NULL;
    int totdentnames;
    int i = 0;
    char **dnarray;

    XrdFfsDent_names_add(&x, "aaa");
    XrdFfsDent_names_add(&x, "bbb");
    XrdFfsDent_names_add(&x, "ccc");

    XrdFfsDent_names_add(&y, "aaa");
    XrdFfsDent_names_add(&y, "aa");
    XrdFfsDent_names_add(&y, "bb");

    XrdFfsDent_names_add(&z, "xxx");

    XrdFfsDent_names_join(&z, &y);
    XrdFfsDent_names_join(&x, &z);

    totdentnames = XrdFfsDent_names_extract(&x, &dnarray);
    char *last, *name;
    for (i=0; i<totdentnames; i++)
    {
        if (i==0 || strcmp(last, dnarray[i]) != 0)
        { 
            name = strdup(dnarray[i]);
            printf(" :== %s\n", name);
            free(name);
            last = dnarray[i];
        }
    }

    XrdFfsDent_cache_init();
    XrdFfsDent_cache_fill("/opt", &dnarray, totdentnames);
    printf("searching /opt/aa : %d\n", XrdFfsDent_cache_search("/opt", "aa"));
    printf("searching /opm/aa : %d\n", XrdFfsDent_cache_search("/opm", "aa"));
    printf("searching /opt/dd : %d\n", XrdFfsDent_cache_search("/opt", "dd"));
    sleep(3);
    printf("searching /opt/aa : %d\n", XrdFfsDent_cache_search("/opt", "aa"));
    sleep(3);
    printf("searching /opt/aa : %d\n", XrdFfsDent_cache_search("/opt", "aa"));
    XrdFfsDent_cache_destroy();

    XrdFfsDent_cache_init();
    i = XrdFfsDent_cache_fill("/opt0", &dnarray, totdentnames);
    sleep(7);
    i = XrdFfsDent_cache_fill("/opt1", &dnarray, totdentnames);
    i = XrdFfsDent_cache_fill("/opt2", &dnarray, totdentnames);
    i = XrdFfsDent_cache_fill("/opt3", &dnarray, totdentnames);
    i = XrdFfsDent_cache_fill("/opt4", &dnarray, totdentnames);
    i = XrdFfsDent_cache_fill("/opt5", &dnarray, totdentnames);
    
    printf("searching /opt0/aa : %d\n", XrdFfsDent_cache_search("/opt0","aa"));
    printf("searching /opt1/aa : %d\n", XrdFfsDent_cache_search("/opt1","aa"));
    printf("searching /opt2/aa : %d\n", XrdFfsDent_cache_search("/opt2","aa"));
    printf("searching /opt3/aa : %d\n", XrdFfsDent_cache_search("/opt3","aa"));
    printf("searching /opt4/aa : %d\n", XrdFfsDent_cache_search("/opt4","aa"));
    printf("searching /opt5/aa : %d\n", XrdFfsDent_cache_search("/opt5","aa"));
    XrdFfsDent_cache_destroy();
    for (i=0; i<totdentnames; i++)
    {
        free(dnarray[i]);
    }
    free(dnarray);
    exit(0);
}
*/

#ifdef __cplusplus
  }
#endif
