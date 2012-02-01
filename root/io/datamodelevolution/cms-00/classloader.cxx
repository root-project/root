//====================================================================
//        Root Database factories implementation
//--------------------------------------------------------------------
//
//        Package    : POOLCore/RootDb (The POOL project)
//
//        Author     : M.Frank
//====================================================================

#ifndef __CINT__
#include "Cintex/Cintex.h"
#endif
#include "Reflex/Reflex.h"
#include "Reflex/PluginService.h"
#include "Reflex/Builder/CollectionProxy.h"

#include "TROOT.h"
#include "TClass.h"
#include "TRealData.h"
#include "TClassEdit.h"
#include "TDataMember.h"

#include "TCollectionProxyFactory.h"
#ifndef __CINT__
#include "Api.h"
#endif

#include "classloader.h"
#include <iostream>

typedef Reflex::Type TypeH;

// using namespace ROOT::Cintex;
// using namespace pool;

void patchClass(TClass* cl, const std::type_info& ) {
   if (gDebug) cerr << "Pool would have patched (from info): " << cl->GetName() << std::endl;
   //assert(0);
} 
void patchClass(TClass* cl, TypeH& ) {
   if (gDebug) cerr << "Pool would have patched (from Reflex type):  " << cl->GetName() << std::endl;
   //assert(0);
}

#include <utility>
#include <vector>

#ifndef __CINT__
namespace ROOT { namespace Cintex  {
  bool IsSTLinternal(const std::string& nam);
  bool IsSTL(const std::string& nam);
  bool IsSTLext(const std::string& nam);
  bool IsTypeOf(TypeH& typ, const std::string& base_name);
  TypeH CleanType(const TypeH& t);
  /// Retrieve CINT class name (in Type.cpp)
  std::string CintName(const TypeH&);
  std::string CintName(const std::string&);
}}

namespace {
  struct CallPreserve {
    bool& m_val;
    bool m_save;
    CallPreserve(bool& val) : m_val(val), m_save(val) { m_val = true; }
    ~CallPreserve() { m_val = m_save; }
  };
}
#endif

namespace pool {
  // Small debug function to dump real data members of a given class
  void showRealData(TClass* cl)   {
    TRealData *dm;
    //    DbPrint log("CINT");
    std::ostream &log(std::cout);

    TIter   next(cl->GetListOfRealData());
    while ((dm = (TRealData *) next()))  {
       log // << DbPrintLvl::Always
        << std::setw(16) << std::left << "showRealdata>"
        << cl->GetName() << "::" << dm->GetName() 
          << " tag:" // << cl->GetClassInfo()->Tagnum() 
        << " Member:"  << (void*)dm 
        << ","  << (void*)dm->GetStreamer()
        << "," << (void*)dm->GetDataMember()
        << ", " << dm->GetDataMember()->GetFullTypeName()
        << " Offset:" << dm->GetThisOffset()
          //        << DbPrint::endmsg;
        << '\n';
    }
  }
  // Small debug function to dump real data members of a given class
  void showDataMembers(TClass* cl)   {
    TDataMember *dm;
    //    DbPrint log("CINT");
    std::ostream &log(std::cout);

    TIter   next(cl->GetListOfDataMembers());
    while ((dm = (TDataMember*) next()))  {
       log //<< DbPrintLvl::Always
        << std::setw(16) << std::left << "showDataMembers>"
        << cl->GetName() << "::" << dm->GetName() 
          << " tag:" // << cl->GetClassInfo()->Tagnum() 
        << " Member:"  << (void*)dm 
        << ","  << dm->GetTypeName()
        << "," << (void*)dm->GetClass()
        << ", " << dm->GetFullTypeName()
        << '\n'; // DbPrint::endmsg;
    }
  }
#if 0
  /// get access to the user supplied streamer function (if any)
  ICINTStreamer* streamer(TypeH& cl)  {
    // First look for custom streamers; 
    // if there are none, check for standard streamers
    if ( cl )   {
      if ( !cl.IsFundamental() )  {
        std::string typ = DbReflex::fullTypeName(cl);
        if ( typ.substr(0,2) == "::" ) typ.replace(0,2,"");
        std::string nam = "CINTStreamer<"+typ+">";
        size_t occ;
        // Replace namespace "::" with "__"
        while ( (occ=nam.find("::")) != std::string::npos )    {
          nam.replace(occ, 2, "__");
        }
        try {
          return Reflex::PluginService::Create<ICINTStreamer*>(nam,cl);
        }
        catch(...) {
          return 0;
        }
      }
    }
    return 0;
  }
#endif
#if 0
  IIOHandler* ioHandler(TypeH& cl)  {
    size_t idx;
    std::string typ = cl.Name(Reflex::SCOPED); // DbReflex::fullTypeName(cl);
    if ( typ.substr(0,2) == "::" ) typ.replace(0,2,"");
    std::string nam = "IIOHandler<"+typ+">";
    // Replace namespace "::" with "__"
    while ( (idx=nam.find("::")) != std::string::npos )    {
      nam.replace(idx, 2, "__");
    }
    try {
      return Reflex::PluginService::Create<IIOHandler*>(nam,cl);
    }
    catch(...) {
      return 0;
    }
  }
#endif

  /// Access streamer info from a void (polymorph) pointer
  TClass* accessType(const TClass* cl, const void* /* ptr */)  {
    return (TClass*)cl;
  }

  TClass* createClass( TypeH typ, ROOT::TGenericClassInfo* info)  {
    TClass* root_class = 0;
    std::string name = typ.Name(Reflex::SCOPED); // DbReflex::fullTypeName(typ);
    std::string nam = name;
    if ( nam.find("stdext::hash_") != std::string::npos )
      nam.replace(3,10,"::");
    if ( nam.find("__gnu_cxx::hash_") != std::string::npos )
      nam.replace(0,16,"std::");
    int kind = TClassEdit::IsSTLCont(nam.c_str());
    if ( kind < 0 ) kind = -kind;

#if 0
    const char* tagname = name.c_str();
    int tagnum = ::G__defined_tagname(tagname, 2);
    G__ClassInfo cl_info(tagnum);
    if ( cl_info.IsValid() )  {
      // Not: these strings MUST be static!
      //if ( !cl_info.ImpFile()    ) cl_info.SetImpFile((char*)cl_info.Name());
      //if (  cl_info.ImpLine()<=0 ) cl_info.SetImpLine(1);
      //if ( !cl_info.DefFile()    ) cl_info.SetDefFile((char*)cl_info.Name());
      //if (  cl_info.DefLine()<=0 ) cl_info.SetDefLine(1);
      switch(kind)  {
        case TClassEdit::kVector:
        case TClassEdit::kList:
        case TClassEdit::kDeque:
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
        case TClassEdit::kSet:
        case TClassEdit::kMultiSet:
          //cl_info.SetVersion(4);
          break;
        case TClassEdit::kNotSTL:
        case TClassEdit::kEnd:
          //cl_info.SetVersion(1);
          break;
      }
    }
#endif

    const std::type_info& tid = typ.TypeInfo();
    bool cl_created = false;
    //std::cout << "Create class " << name << std::endl;
    if ( 0 == root_class )  {
      root_class = info->GetClass();
      cl_created = true;
    }
    else if ( !root_class->GetTypeInfo() )  {
      gROOT->RemoveClass(root_class);
      patchClass(root_class, tid);
      gROOT->AddClass(root_class);
    }
    if ( 0 != root_class )   {
      root_class->Size();
      if ( !typ.IsVirtual() ) root_class->SetGlobalIsA(accessType);
      switch(kind)  {
        case TClassEdit::kVector:
        case TClassEdit::kList:
        case TClassEdit::kDeque:
        case TClassEdit::kMap:
        case TClassEdit::kMultiMap:
        case TClassEdit::kSet:
        case TClassEdit::kMultiSet:
          {
            patchClass(root_class, typ);
            Reflex::Member method = typ.MemberByName("createCollFuncTable");
            if ( !method )   {
              std::cout << name << "' Setup failed to create this class! "
                << "The function createCollFuncTable is not availible."
                << std::endl;
              return 0;
            }
#if ROOT_VERSION_CODE < ROOT_VERSION(5,21,3)
            std::auto_ptr<Reflex::CollFuncTable> m((Reflex::CollFuncTable*)method.Invoke().Address());
#else
            Reflex::CollFuncTable *m = 0;
            method.Invoke(m);
            std::auto_ptr<Reflex::CollFuncTable> auto_delete_func_table(m);
#endif
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,22,0)
            ROOT::TCollectionProxyInfo proxy(tid,
					     m->iter_size,
					     m->value_diff,
					     m->value_offset,
					     m->size_func,
					     m->resize_func,
					     m->clear_func,
					     m->first_func,
					     m->next_func,
					     m->construct_func,
					     m->destruct_func,
					     m->feed_func,
					     m->collect_func,
					     m->create_env);
            root_class->SetCollectionProxy(proxy);
#elif ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
            ROOT::TCollectionProxyInfo proxy(tid,
                                 m->iter_size,
                                 m->value_diff,
                                 m->value_offset,
                                 m->size_func,
                                 m->resize_func,
                                 m->clear_func,
                                 m->first_func,
                                 m->next_func,
                                 m->construct_func,
                                 m->destruct_func,
                                 m->feed_func,
                                 m->collect_func);
            root_class->SetCollectionProxy(proxy);
#else
            std::auto_ptr<TCollectionProxy::Proxy_t> proxy(
  #if ROOT_VERSION_CODE >= ROOT_VERSION(5,4,0)
              TCollectionProxy::GenExplicitProxy(
  #else
              TCollectionProxy::genExplicitProxy(
  #endif
                                 tid,
                                 m->iter_size,
                                 m->value_diff,
                                 m->value_offset,
                                 m->size_func,
                                 m->resize_func,
                                 m->clear_func,
                                 m->first_func,
                                 m->next_func,
                                 m->construct_func,
                                 m->destruct_func,
                                 m->feed_func,
                                 m->collect_func));
            root_class->CopyCollectionProxy(*proxy.get());
#endif

            TClassStreamer* str =
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,15,0)
            TCollectionProxyFactory::GenExplicitClassStreamer(proxy,root_class);
#else
  #if ROOT_VERSION_CODE >= ROOT_VERSION(5,4,0)
              TCollectionProxy::GenExplicitClassStreamer(
  #else
              TCollectionProxy::genExplicitClassStreamer(
  #endif
              tid,
              m->iter_size,
              m->value_diff,
              m->value_offset,
              m->size_func,
              m->resize_func,
              m->clear_func,
              m->first_func,
              m->next_func,
              m->construct_func,
              m->destruct_func,
              m->feed_func,
              m->collect_func
              );
#endif
            root_class->AdoptStreamer(str);
            root_class->SetBit(TClass::kIsForeign);
          }
          break;
        case TClassEdit::kNotSTL:
        case TClassEdit::kEnd:
        default:
          if (gDebug) std::cout << "POOL would have registered its own streamer for " << root_class->GetName() << ".\n";
#if 0
          if ( IsTypeOf(typ,"pool::Reference") )  {
            // std::cout << "patch class:" << typ.Name(SCOPED) << std::endl;
            root_class->AdoptStreamer(new RefStreamer(typ, root_class, 1));
          }
          else if ( IsTypeOf(typ,"pool::Token") )  {
            // std::cout << "patch class:" << typ.Name(SCOPED) << std::endl;
            root_class->AdoptStreamer(new RefStreamer(typ, root_class, 2));
          }
          else   {
            ICINTStreamer* s = streamer(typ);
            IIOHandler*    h = ioHandler(typ);
            if ( s && h )
              root_class->AdoptStreamer(new IOHandlerCustomStreamer(typ, root_class, h, s));
            else if ( s )
              root_class->AdoptStreamer(new CustomStreamer(typ, root_class, s));
            else if ( h )
              root_class->AdoptStreamer(new IOHandlerStreamer(typ, root_class, h));
            else
              root_class->SetBit(TClass::kIsForeign);
          }
#endif
      }
    }
    return root_class;
  }
}

namespace pool {

/// Standard constructor
RootClassLoader::RootClassLoader() 
: m_isConverting(false), m_ignoreLoad(false)
{
  static bool first = true;
  if ( first )  {
    first = false;
    m_ignoreLoad = true;
    // Need to preload STL dlls into ROOT to avoid mismatches in the 
    // Collection proxies.
    //TInterpreter::Instance();
    gROOT->ProcessLine("#include <vector>");
    gROOT->ProcessLine("#include <list>");
    gROOT->ProcessLine("#include <map>");
    gROOT->ProcessLine("#include <set>");
    m_ignoreLoad = false;
  }
  ROOT::GetROOT()->AddClassGenerator(this);
  ROOT::Cintex::Cintex::SetROOTCreator(createClass);
  //Cintex::setDebug(2);
  //Cintex::SetPropagateClassTypedefs(false);
  ROOT::Cintex::Cintex::Enable();
}

/// Standard destructor
RootClassLoader::~RootClassLoader() {
}

/// Release resource
void RootClassLoader::release() {
}

/// Load the native class unconditionally
DbStatus RootClassLoader::loadNativeClass(const std::string& name)  {
  return loadClass(name);
}


/// Load the class unconditionally
DbStatus RootClassLoader::loadClass(const std::string& name)  {
  CallPreserve preserve(m_isConverting);
  TypeH rcl = Reflex::Type::ByName(name); // DbReflex::forTypeName(name);
  TClass* cl = 0;
  bool load_attempted = false;
  if ( !rcl )  {
    cl = i_loadClass(name.c_str());
    load_attempted = true;
    if ( cl )  {
       rcl = Reflex::Type::ByName(cl->GetName()); //DbReflex::forTypeName(cl->GetName());
      if ( !rcl && cl->GetTypeInfo() )  {
         rcl = Reflex::Type::ByTypeInfo(*cl->GetTypeInfo()); // DbReflex::forTypeInfo(*cl->GetTypeInfo());
      }
    }
  }
  if ( rcl )  {
    // First look in internal cache.
#if 0
    IClassHandler* hnd = handler(DbReflex::fullTypeName(rcl), false);
    if ( 0 != hnd )  {
      cl = (TClass*)hnd->nativeClass();
      if ( 0 != cl )  {
        return Success;
      }
    }
    else
#endif
  {
      if ( !cl && !load_attempted ) {
        cl = i_loadClass(name.c_str());
      }
//       if ( cl )  {
//         // std::cout << "Loading class:" << name << std::endl;
//         return setHandler(name, cl->GetName(), new DbClassHandler(rcl, cl));
//       }
    }
  }
//   DbPrint log(  "RootClassLoader");
//   log << DbPrintLvl::Info
//     << "Failed to load dictionary for native class: \"" << name << "\""
//     << DbPrint::endmsg;
  return pool::Error;
}

TClass* RootClassLoader::i_loadClass(const char* classname)  {
  if ( m_ignoreLoad )   {
    return 0;
  }
  else  {
    if (gDebug) std::cout << "Pool loading: " << classname << '\n';
    switch(classname[0])  {
      case 'b':
        if ( strcmp(classname,"bool")           ==0 ) return 0;
        break;
      case 'l':
        if ( strncmp(classname,"lib",3)         ==0 ) return 0;
        if ( strcmp(classname,"long")           ==0 ) return 0;
        if ( strcmp(classname,"long long")      ==0 ) return 0;
        if ( strcmp(classname,"long long int")  ==0 ) return 0;
        break;
      case 'L':
        if ( strcmp(classname,"Long_t")         ==0 ) return 0;
        if ( strcmp(classname,"Long64_t")       ==0 ) return 0;
        break;
      case 'i':
        if ( strcmp(classname,"int")            ==0 ) return 0;
        if ( strcmp(classname,"__int64")        ==0 ) return 0;
        break;
      case 'I':
        if ( strcmp(classname,"Int_t")          ==0 ) return 0;
        break;
      case 'e':
        if ( strncmp(classname,"enum ",5)       ==0 ) return 0;
        break;
      case 'd':
        if ( strcmp(classname,"double")         ==0 ) return 0;
        break;
      case 'D':
        if ( strcmp(classname,"Double_t")       ==0 ) return 0;
        break;
      case 'f':
        if ( strcmp(classname,"float")          ==0 ) return 0;
        break;
      case 'F':
        if ( strcmp(classname,"Float_t")        ==0 ) return 0;
        break;
      case 's':
        if ( strcmp(classname,"short")          ==0 ) return 0;
        break;
      case 'S':
        if ( strcmp(classname,"Short_t")        ==0 ) return 0;
        break;
      case 'u':
        if ( strncmp(classname,"unknown",7)     ==0 ) return 0;
        if ( strcmp(classname,"unsigned int")   ==0 ) return 0;
        if ( strcmp(classname,"unsigned short") ==0 ) return 0;
        if ( strcmp(classname,"unsigned long")  ==0 ) return 0;
        if ( strcmp(classname,"unsigned char")  ==0 ) return 0;
        if ( strcmp(classname,"unsigned long long")      ==0 ) return 0;
        if ( strcmp(classname,"unsigned long long int")  ==0 ) return 0;
        break;
    }
    size_t len = ::strlen(classname);
    if ( len>1 && classname[len-1]=='*' ) return 0;

    TypeH rcl = Reflex::Type::ByName(classname); // DbReflex::forTypeName(classname);
    if ( rcl && (rcl.IsFundamental() || rcl.IsPointer()) )  {
      return 0;
    } 
    TClass* cl = ROOT::GetROOT()->GetClass(classname, kTRUE);
    if ( !cl ) {
      cl = TClass::GetClass( TClassEdit::ResolveTypedef(classname).c_str()); 
           // CINT::Typedefs::apply(classname);
      if ( !cl )  {
        size_t idx;
        std::string root_name = ROOT::Cintex::CintName(classname);
        cl = ROOT::GetROOT()->GetClass(root_name.c_str(), kTRUE);
	if ( !cl ) {
	  while( (idx=root_name.find("const ")) != std::string::npos )
	    root_name.replace(idx, 6, "");
	  cl = ROOT::GetROOT()->GetClass(root_name.c_str(), kTRUE);
	}
      }
    }
    return cl;
  }
  return 0;
}

/// Load the class unconditionally
DbStatus RootClassLoader::unloadClass(const std::string& name)  {
  std::cout << "Unloading of " << name << " not done\n";
//   DbPrint log(  "RootClassLoader");
//   log << DbPrintLvl::Error << "Unloading of reflection gateways currently not"
//     << " supported... Class: \"" << name << "\"" << DbPrint::endmsg;
  return pool::Error;
}

/// Overloading TClassGenerator
TClass* RootClassLoader::GetClass(const char* classname, Bool_t load) {
  if ( !m_isConverting && load )  {
    CallPreserve preserve(m_isConverting);
    return i_loadClass(classname);
  }
  return 0;
}

/// Overloading TClassGenerator
TClass* RootClassLoader::GetClass(const type_info& typeinfo, Bool_t load) {
  if ( !m_isConverting && load )  {
    CallPreserve preserve(m_isConverting);
    TypeH lcg_cl = Reflex::Type::ByTypeInfo(typeinfo); // DbReflex::forTypeInfo(typeinfo);
    if ( lcg_cl )  {
      if ( loadNativeClass(lcg_cl.Name(Reflex::SCOPED)) ) { // .isSuccess() )  {
        TClass* cl = ROOT::GetROOT()->GetClass(typeinfo, kTRUE);
        return cl;
      }
    }
  }
  return 0;
}

}
