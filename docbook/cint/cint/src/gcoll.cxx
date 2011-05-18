/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file gcoll.c
 ************************************************************************
 * Description:
 *  Garbage Collection library and dictionary rewinding
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

#define G__IMMEDIATE_GARBAGECOLLECTION

/**************************************************************************
* Garbage collection library usage
*
*  G__init_garbagecollection();
*
*    T *a = malloc(sizeof(T));
*  G__add_alloctable(a,'u',G__defined_tagnum("T"));
*  G__add_refcount(a,&a);
*
*    T *b=a;
*  G__add_refcount(b,&b);
*
*  G__del_refcount(b,&b);
*    b=NULL;
*
*  G__del_alloctable(a);
*    free((void*)a);
*
*  G__garbagecollection();
*
*
**************************************************************************/



#define G__MAXREFTABLE 5
#define G__MAXALLOCTABLE 50

struct G__reflist {
  void **ref;
  struct G__reflist *prev;
  struct G__reflist *next;
};

struct G__alloclist {
  void *allocedmem;
  char type;
  short tagnum;
  struct G__reflist *reflist;
  struct G__alloclist *prev;
  struct G__alloclist *next;
};

static struct G__alloclist *G__alloctable;
static struct G__alloclist *G__p_alloc;
static unsigned int G__count_garbagecollection;

/**************************************************************************
* Static functions
*
*
**************************************************************************/

/**************************************************************************
* G__search_alloctable()
**************************************************************************/
static struct G__alloclist* G__search_alloctable(void *mem)
{
  struct G__alloclist *alloc;
  alloc = G__alloctable;
  while(alloc) {
    if(mem==alloc->allocedmem) return(alloc);
    alloc=alloc->next;
  }
  return((struct G__alloclist*)NULL);
}

/**************************************************************************
* G__free_reflist()
**************************************************************************/
static void G__free_reflist(G__reflist *reflist)
{
  if(reflist) {
    if(reflist->next) {
      G__free_reflist(reflist->next);
    }
    if(reflist->ref) *reflist->ref=(void*)NULL;
    free((void*)reflist);
  }
}

/**************************************************************************
* G__delete_reflist()
*
*  1) delete entry from alloclist
*
**************************************************************************/
static struct G__reflist* G__delete_reflist(G__alloclist *alloc,G__reflist *reflist)
{
  struct G__reflist *freed;
  static struct G__reflist temp;
  
  if(reflist->prev) {
    freed=reflist;
    reflist->prev->next = reflist->next;
    if(reflist->next) reflist->next->prev=reflist->prev;
    reflist = reflist->prev;
  }
  else {
    freed=reflist;
    alloc->reflist = reflist->next;
    if(alloc->reflist) alloc->reflist->prev=(struct G__reflist*)NULL;
    temp.next = alloc->reflist;
    reflist = &temp;
  }
  free((void*)freed);
  
  return(reflist);
}

/**************************************************************************
* G__delete_alloctable()
*
*  1) delete entry from alloclist
*
**************************************************************************/
static struct G__alloclist* G__delete_alloctable(G__alloclist *alloc)
{
  struct G__alloclist *freed;
  static struct G__alloclist temp;

  if(!alloc->next) { 
    G__p_alloc = alloc->prev;
  }

  if(alloc->prev) {
    freed=alloc;
    alloc->prev->next = alloc->next;
    if(alloc->next) alloc->next->prev=alloc->prev;
    alloc = alloc->prev;
  }
  else {
    freed=alloc;
    G__alloctable = alloc->next;
    if(G__alloctable) G__alloctable->prev=(struct G__alloclist*)NULL;
    temp.next = G__alloctable;
    alloc = &temp;
  }
  free((void*)freed);

  return(alloc);
}

/**************************************************************************
* G__destroy_garbageobject()
*
*  1) If class object call destructor
*  2) If interpreted class or fundamental class, free memory or close file
*
**************************************************************************/
void G__destroy_garbageobject(G__alloclist *alloc)
{
  long store_tagnum;
  long store_struct_offset;
  long store_globalvarpointer;
  int done=0;
  G__FastAllocString dtor(G__ONELINE);

  if(-1!=alloc->tagnum) {
    /* Call destructor if class object */
     dtor.Format("~%s()",G__struct.name[alloc->tagnum]);
    store_globalvarpointer = G__globalvarpointer;
    store_tagnum = G__tagnum;
    store_struct_offset = G__store_struct_offset;
    G__tagnum = alloc->tagnum;
    G__store_struct_offset = (long)alloc->allocedmem;
    if(G__CPPLINK==G__struct.iscpplink[alloc->tagnum]) {
      G__globalvarpointer = G__store_struct_offset;
    }
    else {
      G__globalvarpointer = G__PVOID;
    }
    G__getfunction(dtor,&done ,G__TRYDESTRUCTOR);
    G__tagnum = store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__globalvarpointer = store_globalvarpointer;
    if(G__CPPLINK!=G__struct.iscpplink[alloc->tagnum]) {
      /* free resource , interpreted class */
      free((void*)alloc->allocedmem);
    }
  }
  else {
    /* Fundamental type, free resource */
    if('E'==alloc->type) {
      fclose((FILE*)alloc->allocedmem);
    }
    else {
      free((void*)alloc->allocedmem);
    }
  }

  /* delete entry from alloclist */
  /* alloc=G__delete_alloctable(alloc); */

  /* increment garbage collection count */
  ++G__count_garbagecollection;
}


/**************************************************************************
* Exported functions
*
*
**************************************************************************/

/**************************************************************************
* G__init_garbagecollection()
**************************************************************************/
void G__init_garbagecollection()
{
  static int state=1;
  if(state) {
    G__alloctable=(struct G__alloclist*)NULL;;
    G__p_alloc = G__alloctable;
    G__count_garbagecollection=0;
    state=0;
  }
}

/**************************************************************************
* G__garbagecollection()
**************************************************************************/
int G__garbagecollection()
{
  struct G__alloclist *alloc;
  struct G__reflist *reflist;
  unsigned int count;

#ifndef G__IMMEDIATE_GARBAGECOLLECTION
  G__fprinterr(G__serr,"!!! Reference Count Control start !!!\n");
#endif

  alloc = G__alloctable;
  while(alloc) {
    if(!alloc->reflist) {
      /* This object has no reference, means stray pointer */
      G__destroy_garbageobject(alloc);
      alloc=G__delete_alloctable(alloc);
    }
    else {
      /* Delete dummy reference count for returned pointer */
      reflist = alloc->reflist;
      while(reflist) {
        if((void**)NULL==reflist->ref) {
          reflist=G__delete_reflist(alloc,reflist);
        }
        reflist=reflist->next;
      }
    }
    alloc=alloc->next;
  }

  G__fprinterr(G__serr,"!!! %d object(s) deleted by Reference Count Control !!!\n"
          ,G__count_garbagecollection);
  count = G__count_garbagecollection;
  G__count_garbagecollection=0;

  return(count);
}


/**************************************************************************
* G__add_alloctable()
**************************************************************************/
void G__add_alloctable(void *allocedmem,int type,int tagnum)
{
  if(G__p_alloc) {
    G__p_alloc->next=(struct G__alloclist*)malloc(sizeof(struct G__alloclist));
    G__p_alloc->next->prev = G__p_alloc;
    G__p_alloc=G__p_alloc->next;
  }
  else {
    G__alloctable=(struct G__alloclist*)malloc(sizeof(struct G__alloclist));
    G__p_alloc = G__alloctable;
    G__p_alloc->prev =(struct G__alloclist*)NULL;
  }
  G__p_alloc->allocedmem = allocedmem;
  G__p_alloc->type = (char)type;
  G__p_alloc->tagnum = (short) tagnum;
  G__p_alloc->reflist=(struct G__reflist*)NULL;
  G__p_alloc->next=(struct G__alloclist*)NULL;
}

/**************************************************************************
* G__del_alloctable()
**************************************************************************/
int G__del_alloctable(void *allocmem)
{
  struct G__alloclist *alloc;

  alloc = G__search_alloctable(allocmem);
  if(alloc) {
    G__free_reflist(alloc->reflist);
    G__delete_alloctable(alloc);
    return(0);
  }
  else {
    G__fprinterr(G__serr,"Error: Can not free 0x%lx, not allocated."
            ,(long)allocmem);
    G__genericerror((char*)NULL);
    return(1);
  }
}

/**************************************************************************
* G__add_refcount()
**************************************************************************/
int G__add_refcount(void *allocedmem,void **storedmem)
{
  struct G__alloclist *alloc;
  struct G__reflist *reflist;
  alloc = G__search_alloctable(allocedmem);
  if(alloc) {
    if(alloc->reflist) {
      reflist=alloc->reflist;
      while(reflist->next) reflist=reflist->next;
      reflist->next=(struct G__reflist*)malloc(sizeof(struct G__reflist));
      reflist->next->prev=reflist;
      reflist=reflist->next;
      reflist->next=(struct G__reflist*)NULL;
      reflist->ref=storedmem;
    }
    else {
      alloc->reflist=(struct G__reflist*)malloc(sizeof(struct G__reflist));
      reflist = alloc->reflist;
      reflist->prev=(struct G__reflist*)NULL;
      reflist->next=(struct G__reflist*)NULL;
      reflist->ref=storedmem;
    }
  }
  return(0);
}

/**************************************************************************
* G__del_refcount()
**************************************************************************/
int G__del_refcount(void *allocedmem,void **storedmem)
{
  int flag=1;
  struct G__alloclist *alloc;
  struct G__reflist *reflist;
  alloc = G__search_alloctable(allocedmem);
  if(alloc) {
    reflist = alloc->reflist;
    while(reflist) {
      if(reflist->ref==storedmem) {
        reflist=G__delete_reflist(alloc,reflist);
      }
      else if((void**)NULL==reflist->ref) {
        reflist=G__delete_reflist(alloc,reflist);
        flag=0;
      }
      reflist=reflist->next;
    }
#ifdef G__IMMEDIATE_GARBAGECOLLECTION
    if(!alloc->reflist && flag) {
#ifdef G__DEBUG
      G__fprinterr(G__serr,"!!! %s object deleted by Reference Count Control !!!\n"
              ,G__type2string(alloc->type,alloc->tagnum,-1,0,0));
#endif
      G__destroy_garbageobject(alloc);
      G__delete_alloctable(alloc);
    }
#endif
  }
  return(0);
}


/**************************************************************************
* Debugging functions
*
*
**************************************************************************/

/**************************************************************************
* G__disp_garbagecollection()
**************************************************************************/
int G__disp_garbagecollection(FILE *fout)
{
  static struct G__alloclist *alloc;
  struct G__reflist *reflist;

  alloc = G__alloctable;
  fprintf(fout,"Allocated memory =========================================\n");
  fprintf(fout,"type                : location   : reference(s)\n");
  while(alloc) {
    fprintf(fout,"%-20s: 0x%lx :"
            ,G__type2string(alloc->type,alloc->tagnum,-1,0,0)
            ,(long)alloc->allocedmem);
    reflist = alloc->reflist;
    while(reflist) {
      fprintf(fout," 0x%lx ,",(long)reflist->ref);
      reflist = reflist->next;
    }
    fprintf(fout,"\n");
    alloc=alloc->next;
  }
  return(0);
}

} /* extern "C" */

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
