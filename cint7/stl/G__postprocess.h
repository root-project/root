/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*************************************************************************
 * stl/G__postprocess.h
 *
 * Description:
 *  This source code should be used to disable specific member function 
 *  linkage from STL container.
 *
 *  #pragma link postprocess G__postprocess.cxx G__postprocess("Ary");
 *
 *************************************************************************/
#include <ertti>

void G__postprocess(const char* templatename) {
  printf("postprocessing linkdefs\n");
  char classname[200],tempname[200];
  G__ClassInfo c;
  G__ClassInfo global;
  G__TypeInfo tmparg;
  long offset;
  int len;
  
  sprintf(tempname,"%s<",templatename);
  len = strlen(tempname);
  
  // iterate on all class, struct
  while(c.Next()) { 
    strcpy(classname,c.Name());
    
    // choose instantiated template class and interpreted one
    if((0==c.Property()&G__BIT_ISCOMPILED) && 
       strncmp(classname,tempname,len)==0) {
      
      // Assuming the first tmparg is the element type, get that typename
      char *tmpargname = strchr(classname,'<');
      char *p2 = strrchr(classname,'>');
      char *p3 = strrchr(classname,',');
      if(!tmpargname || !p2) continue;
      if(p2) *p2 = 0;
      if(p3) *p3 = 0;
      ++tmpargname;
      
      // Get type information of template argument
      tmparg.Init(tmpargname);
      if(tmparg.IsValid()) {
	if(tmparg.Property()&G__BIT_ISPOINTER) {
	  // activate all member function if temp arg is a fundamental type
	  // or pointer of something
	  G__controloperation("operator+",c,tmparg,0);
	  G__controloperation("operator-",c,tmparg,0);
	  G__controloperation("operator*",c,tmparg,0);
	  G__controloperation("operator/",c,tmparg,0);
	  G__controloperation("operator%",c,tmparg,0);
	  G__controloperation("operator&",c,tmparg,0);
	  G__controloperation("operator|",c,tmparg,0);
	  G__controloperation("operator&&",c,tmparg,0);
	  G__controloperation("operator||",c,tmparg,0);
	  G__controloperation("operator<<",c,tmparg,0);
	  G__controloperation("operator>>",c,tmparg,0);
	  G__controloperation("operator<=",c,tmparg,0);
	  G__controloperation("operator>=",c,tmparg,0);
	  G__controloperation("operator<",c,tmparg,0);
	  G__controloperation("operator>",c,tmparg,0);
	  G__controloperation("operator==",c,tmparg,1);
	  G__controloperation("operator!=",c,tmparg,1);
	}
	else if(tmparg.Property()&G__BIT_ISFUNDAMENTAL) {
	  if(strcmp(tmparg.TrueName(),"double")==0 ||
	       strcmp(tmparg.TrueName(),"float")==0) {
	    G__controloperation("operator+",c,tmparg,1);
	    G__controloperation("operator-",c,tmparg,1);
	    G__controloperation("operator*",c,tmparg,1);
	    G__controloperation("operator/",c,tmparg,1);
	    G__controloperation("operator%",c,tmparg,0);
	    G__controloperation("operator&",c,tmparg,0);
	    G__controloperation("operator|",c,tmparg,0);
	    G__controloperation("operator&&",c,tmparg,0);
	    G__controloperation("operator||",c,tmparg,0);
	    G__controloperation("operator<<",c,tmparg,0);
	    G__controloperation("operator>>",c,tmparg,0);
	    G__controloperation("operator<=",c,tmparg,1);
	    G__controloperation("operator>=",c,tmparg,1);
	    G__controloperation("operator<",c,tmparg,1);
	    G__controloperation("operator>",c,tmparg,1);
	    G__controloperation("operator==",c,tmparg,1);
	    G__controloperation("operator!=",c,tmparg,1);
	  }
	  else {
	    G__controloperation("operator+",c,tmparg,1);
	    G__controloperation("operator-",c,tmparg,1);
	    G__controloperation("operator*",c,tmparg,1);
	    G__controloperation("operator/",c,tmparg,1);
	    G__controloperation("operator%",c,tmparg,1);
	    G__controloperation("operator&",c,tmparg,1);
	    G__controloperation("operator|",c,tmparg,1);
	    G__controloperation("operator&&",c,tmparg,1);
	    G__controloperation("operator||",c,tmparg,1);
	    G__controloperation("operator<<",c,tmparg,1);
	    G__controloperation("operator>>",c,tmparg,1);
	    G__controloperation("operator<=",c,tmparg,1);
	    G__controloperation("operator>=",c,tmparg,1);
	    G__controloperation("operator<",c,tmparg,1);
	    G__controloperation("operator>",c,tmparg,1);
	    G__controloperation("operator==",c,tmparg,1);
	    G__controloperation("operator!=",c,tmparg,1);
	  }
	}
	else if(tmparg.Property()&G__BIT_ISCLASS) {
	  // in case of temp arg is a class
	  G__controloperation("operator+",c,tmparg);
	  G__controloperation("operator-",c,tmparg);
	  G__controloperation("operator*",c,tmparg);
	  G__controloperation("operator/",c,tmparg);
	  G__controloperation("operator%",c,tmparg);
	  G__controloperation("operator&",c,tmparg);
	  G__controloperation("operator|",c,tmparg);
	  G__controloperation("operator&&",c,tmparg);
	  G__controloperation("operator||",c,tmparg);
	  G__controloperation("operator<<",c,tmparg);
	  G__controloperation("operator>>",c,tmparg);
	  G__controloperation("operator<=",c,tmparg);
	  G__controloperation("operator>=",c,tmparg);
	  G__controloperation("operator<",c,tmparg);
	  G__controloperation("operator>",c,tmparg);
	  G__controloperation("operator==",c,tmparg);
	  G__controloperation("operator!=",c,tmparg);
	}
      }
    }
  }
}

void G__controloperation(const char* fname,G__ClassInfo& c
			 ,G__TypeInfo& tmparg,int globalcomp) 
{
  long offset;
  G__MethodInfo m;
  G__ClassInfo global;
  char classname[200],args[300];
  strcpy(classname,c.Name());
  int flag = globalcomp? -1 : 0 ;
  
  // member function
  m = c.GetMethod(fname,classname,&offset);
  m.SetGlobalcomp(flag);
  
  // global function
  sprintf(args,"%s,%s",classname,classname);
  m = global.GetMethod(fname,args,&offset);
  m.SetGlobalcomp(flag);
}
  
void G__controloperation(const char* fname,G__ClassInfo& c
			 ,G__TypeInfo& tmparg) 
{
  long offset;
  G__MethodInfo m;
  G__ClassInfo global;
  int globalcomp = 0;
  char classname[200],args[300];
  strcpy(classname,tmparg.Name());
  
  // member function
  m = tmparg.GetMethod(fname,classname,&offset);
  if(m.IsValid()) globalcomp = 1;
  
  // global function
  sprintf(args,"%s,%s",classname,classname);
  m = global.GetMethod(fname,args,&offset);
  if(m.IsValid()) globalcomp = 1;
  
  G__controloperation(fname,c,tmparg,globalcomp); 
}

/*************************************************************************
* Use this function with '#pragma link postprocess' as follows at the end 
* of ROOT linkdef files.
*
*  #pragma link postprocess G__postprocess.h G__ROOT_T();
*
*************************************************************************/
#include <ertti>

void G__ROOT_T() {
  G__DataMemberInfo global;
  long prop;
  while(global.Next()) {
    prop = global.Property();
    if(0==(prop&(~G__BIT_ISPUBLIC)) &&
       strncmp("ROOT_T",global.Name(),6)==0) {
      global.SetGlobalcomp(-1);
    }
  }
}

