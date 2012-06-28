/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*********************************************************************
* typeinfo.h
*
*  Run time type identification
*
* Memo:
*   typeid(typename) , typeid(expression) is implemented as special 
*  function in the cint body src/G__func.c. 
*
*   As an extention, G__typeid(char *name) is defined in src/G__func.c
*  too for more dynamic use of the typeid.
*
*   type_info is extended to support non-polymorphic type objects.
*
*   In src/G__sizeof.c , G__typeid() is implemented. It relies on
*  specific binary layout of type_info object. If order of type_info
*  member declaration is modified, src/G__sizeof.c must be modified
*  too.
*
*********************************************************************/

#ifdef __CINT__

#ifndef G__TYPEINFO_H
#define G__TYPEINFO_H

#include <bool.h>

/*********************************************************************
* Functions embedded in cint core
* Most of those functions are defined in src/sizeof.c
* 
*********************************************************************/
// type_info typeid(expression);
// type_info typeid(char *typename);
// type_info G__typeid(char *expression);
// long G__get_classinfo(char *item,int tagnum);
// long G__get_variableinfo(char *item,long *handle,long *index,long tagnum);
// long G__get_functioninfo(char *item,long *handle,long &index,long tagnum);


/*********************************************************************
* type_info
*
*  Included in ANSI/ISO resolution proposal 1995 spring
* 
*********************************************************************/
class type_info {
 public:
  virtual ~type_info() { }  // type_info is polymorphic
  bool operator==(const type_info&) const;
  bool operator!=(const type_info&) const;
  bool before(const type_info&) const;

  const char* name() const;

 private:
  type_info(const type_info&);
 protected: // original enhancement
  type_info& operator=(const type_info&);

  // implementation dependent representation
 protected:
  long type;      // intrinsic types
  long tagnum;    // class/struct/union
  long typenum;   // typedefs
  long reftype;   // pointing level and reference types
  long size;      // size of the object
#ifndef G__OLDIMPLEMENTATION1895
  long isconst;   // constness
#endif

 public: // original enhancement
  type_info() { }
};


bool type_info::operator==(const type_info& a) const
{
  if(reftype == a.reftype && tagnum == a.tagnum && type == a.type) 
    return(true);
  else 
    return(false);
}

bool type_info::operator!=(const type_info& a) const
{
  if( *this == a ) return(false);
  else             return(true);
}

bool type_info::before(const type_info& a) const
{
  if(-1!=tagnum) 
    return( tagnum < a.tagnum );
  else if(-1!=a.tagnum) 
    return( -1 < a.tagnum );
  else 
    return( type < a.type );
}

const char* type_info::name() const
{
  static char namestring[100];
  //printf("%d %d %d %d\n",type,tagnum,typenum,reftype);
#ifndef G__OLDIMPLEMENTATION1895
#ifdef G__GNUC
  char *cptr = G__type2string(type,tagnum,typenum,reftype,isconst);
  sprintf(namestring,"%d%s",strlen(cptr),cptr);
#else
  strcpy(namestring,G__type2string(type,tagnum,typenum,reftype,isconst));
#endif
#else
  strcpy(namestring,G__type2string(type,tagnum,typenum,reftype));
#endif
  return(namestring);
}

type_info::type_info(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
#ifndef G__OLDIMPLEMENTATION1895
  isconst = a.isconst;
#endif
}

type_info& type_info::operator=(const type_info& a)
{
  type = a.type;
  tagnum = a.tagnum;
  typenum = a.typenum;
  reftype = a.reftype;
  size = a.size;
#ifndef G__OLDIMPLEMENTATION1895
  isconst = a.isconst;
#endif
  return(*this);
}

/**************************************************************************
* original enhancment
**************************************************************************/
type_info::type_info()
{
  type = 0;
  tagnum = typenum = -1;
  reftype = 0;
#ifndef G__OLDIMPLEMENTATION1895
  isconst = 0;
#endif
}


/**************************************************************************
* Further runtime type checking requirement from Fons Rademaker
**************************************************************************/

/*********************************************************************
* G__class_info
*
*********************************************************************/
class G__class_info : public type_info {
 public:
  G__class_info() { init(); }
  G__class_info(type_info& a) { init(a); }
  G__class_info(char *classname) { init(G__typeid(classname)); }
  
  void init() {
    typenum = -1;
    reftype = 0;
    tagnum = G__get_classinfo("next",-1);
    size = G__get_classinfo("size",tagnum);
    type = G__get_classinfo("type",tagnum);
  }

  void init(type_info& a) {
    type_info *p=this;
    *p = a;
  }

  G__class_info& operator=(G__class_info& a) {
    type = a.type;
    tagnum = a.tagnum;
    typenum = a.typenum;
    reftype = a.reftype;
    size = a.size;
  }

  G__class_info& operator=(type_info& a) {
    init(a);
  }

  G__class_info* next() {
    tagnum=G__get_classinfo("next",tagnum);
    if(-1!=tagnum) return(this);
    else {
      size = type = 0;
      return((G__class_info*)NULL);
    }
  }

  char *title() {
    return((char*)G__get_classinfo("title",tagnum));
  }

  // char *name() is inherited from type_info

  char *baseclass() {
    return((char*)G__get_classinfo("baseclass",tagnum));
  }


  int isabstract() {
    return((int)G__get_classinfo("isabstract",tagnum));
  }

  // can be implemented
  // int iscompiled();

  int Tagnum() {
    return(tagnum);
  }

};
  

/*********************************************************************
* G__variable_info
*
*********************************************************************/
class G__variable_info {
 public:
  G__variable_info() { init(); }
  G__variable_info(G__class_info& a) { init(a); }
  G__variable_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_variableinfo("new",&handle,&index,tagnum=-1);
  }

  void init(G__class_info& a) {
    G__get_variableinfo("new",&handle,&index,tagnum=a.Tagnum());
  }

  G__variable_info* next() {
    if(G__get_variableinfo("next",&handle,&index,tagnum)) return(this);
    else  return((G__variable_info*)NULL);
  }

  char *title() {
    return((char*)G__get_variableinfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_variableinfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_variableinfo("type",&handle,&index,tagnum));
  }

  int offset() {
    return((int)G__get_variableinfo("offset",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};  

/*********************************************************************
* G__function_info
*
*********************************************************************/
class G__function_info {
 public:
  G__function_info() { init(); }
  G__function_info(G__class_info& a) { init(a); }
  G__function_info(char *classname) { init(G__class_info(classname)); }

  void init() {
    G__get_functioninfo("new",&handle,&index,tagnum=-1);
  } // initialize for global function

  void init(G__class_info& a) {
    G__get_functioninfo("new",&handle,&index,tagnum=a.Tagnum());
  } // initialize for member function

  G__function_info* next() {
    if(G__get_functioninfo("next",&handle,&index,tagnum)) return(this);
    else return((G__function_info*)NULL);
  }

  char *title() {
    return((char*)G__get_functioninfo("title",&handle,&index,tagnum));
  }

  char *name() {
    return((char*)G__get_functioninfo("name",&handle,&index,tagnum));
  }

  char *type() {
    return((char*)G__get_functioninfo("type",&handle,&index,tagnum));
  }

  char *arglist() {
    return((char*)G__get_functioninfo("arglist",&handle,&index,tagnum));
  }

  // can be implemented
  // char *access(); // return public,protected,private
  // int isstatic();
  // int iscompiled();
  // int isvirtual();
  // int ispurevirtual();

 private:
  long handle; // pointer to variable table
  long index;
  long tagnum; // class/struct identity
};  

/*********************************************************************
* G__string_buf
*
*  This struct is used as temporary object for returning title strings.
* Size of buf[] limits maximum length of the title string you can
* describe. You can increase size of it here to increase it.
*
*********************************************************************/
struct G__string_buf {
  char buf[256];
};


/*********************************************************************
* Example code
*
*  Following functions are the examples of how to use the type info
* facilities.
*
*********************************************************************/

#ifdef __CINT__
void G__list_class(void) {
  G__class_info a;
  do {
    printf("%s:%s =%d '%s'\n",a.name(),a.baseclass(),a.isabstract(),a.title());
  } while(a.next());
}

void G__list_class(char *classname) {
  G__list_memvar(classname);
  G__list_memfunc(classname);
}

void G__list_memvar(char *classname) {
  G__variable_info a=G__variable_info(G__typeid(classname));
  do {
    printf("%s %s; offset=%d '%s'\n",a.type(),a.name(),a.offset(),a.title());
  } while(a.next());
}

void G__list_memfunc(char *classname) {
  G__function_info a=G__function_info(G__typeid(classname));
  do {
    printf("%s %s(%s) '%s'\n",a.type(),a.name(),a.arglist(),a.title());
  } while(a.next());
}
#endif

#endif /* of G__TYPEINFO_H */

#endif /* __CINT__ */
