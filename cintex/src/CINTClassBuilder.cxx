// @(#)root/cintex:$Name:  $:$Id: CINTClassBuilder.cxx,v 1.4 2005/11/17 14:12:33 roiser Exp $
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Tools.h"
#include "Cintex/Cintex.h"
#include "CINTdefs.h"
#include "CINTClassBuilder.h"
#include "CINTScopeBuilder.h"
#include "CINTFunctionBuilder.h"
#include "CINTFunctional.h"
#include "Api.h"
#include <list>
#include <set>
#include <iomanip>
#include <sstream>

using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

  struct PendingBase {
    Type   basetype;
    int    tagnum;
    size_t offset;
    PendingBase( const Type& t, int n, size_t o) : basetype(t), tagnum(n), offset(o) {}
  };

  class CommentBuffer  {
  private:
    typedef std::vector<char*> VecC;
    VecC fC;
    CommentBuffer() {}
    ~CommentBuffer()  {
      for(VecC::iterator i=fC.begin(); i != fC.end(); ++i)
        delete [] *i;
      fC.clear();
    }
  public:
    static CommentBuffer& Instance()  {     
      static CommentBuffer inst;
      return inst;
    }
    void add(char* cm)  {
      fC.push_back(cm);
    }
  };

  static list<PendingBase>& pendingBases() {
     static list<PendingBase> s_pendingBases;
     return s_pendingBases;
  }  

  CINTClassBuilder& CINTClassBuilder::Get(const Type& cl) {
    CINTClassBuilders& builders = CINTClassBuilders::Instance();
    CINTClassBuilders::iterator it = builders.find(cl);
    if( it != builders.end() )  return *(*it).second;
    CINTClassBuilder* builder = new CINTClassBuilder(cl);
    builders[cl] = builder;
    return *builder;
  }

  CINTClassBuilder::CINTClassBuilder(const Type& cl)
  : fClass(cl), fName(CintName(cl)), fPending(true), 
    fSetup_memvar(0), fSetup_memfunc(0), fBases(0)
  {
    fTaginfo = new G__linked_taginfo;
    fTaginfo->tagnum  = -1;   // >> need to be pre-initialized to be understood by CINT
    fTaginfo->tagtype = 'c';
    fTaginfo->tagname = fName.c_str();
    fTaginfo->tagnum = G__defined_tagname(fTaginfo->tagname, 2);

    if ( fTaginfo->tagnum < 0 )  {
      Setup_tagtable();
    }
    else   {
      G__ClassInfo info(fTaginfo->tagnum);
      if ( !info.IsLoaded() )  {
        Setup_tagtable();
      }
      else  {
        fPending = false;
        if( Cintex::Debug() > 1 ) std::cout << "Precompiled class:" << fName << std::endl;
      }
    }
  }

  CINTClassBuilder::~CINTClassBuilder() {
    delete fTaginfo;
    Free_function((void*)fSetup_memfunc);
    Free_function((void*)fSetup_memvar);
  }

  void CINTClassBuilder::Setup() {
    if ( fPending ) {
      if ( Cintex::Debug() ) std::cout << "Building class " << fName << std::endl;
      fPending = false;

      // Setup_memfunc();        // It is delayed
      // Setup_memvar();         // It is delayed
      // Setup_inheritance();
      Setup_inheritance_simple();
      Setup_typetable();
    }
    return;
  }

  void CINTClassBuilder::Setup_tagtable() {

    // Setup ScopeNth
    Scope ScopeNth = fClass.DeclaringScope();
    if ( ScopeNth ) CINTScopeBuilder::Setup(ScopeNth);
    else {
      ScopeNth = Scope::ByName(Tools::GetScopeName(fClass.Name(SCOPED)));
      if( ScopeNth.Id() ) CINTScopeBuilder::Setup(ScopeNth);
    }

    // Setup tag number
    fTaginfo->tagnum = G__get_linked_tagnum(fTaginfo);
    std::string comment = fClass.Properties().HasKey("comment") ? 
                          fClass.Properties().PropertyAsString("comment").c_str() :
                          "";
    // Assume some minimal class functionality; see below for explanation
    int rootFlag = 0;
    rootFlag   += 0x00020000;         // No operator >> ()
    //if ( 0 != streamer() )    {     // If a customized streamer exists
    //  if ( !(isIntrinsic() || getFlag(ISSTRING)) )  {
    //    rootFlag += 0x00010000;     // Bit 1 Set
    //  }                             //
    //}                               //
    //else  {                         // Automatic schema evolution
    //  rootFlag += 0x00040000;       //
    //}                               //
    if ( fClass.IsAbstract() ) {
      rootFlag += G__BIT_ISABSTRACT;  // Abstract class
    }                                 //
    //  rootFlag += 0x00000100;       // Class has Empty Constructor
    //  rootFlag += 0x00000400;       // Class has destructor
    //if ( 0 != fCopyConstructor ) { //
    //  rootFlag += 0x00000200;       // Class has copy Constructor
    //}                               //
    if ( fClass.HasBase(Type::ByName("TObject")))   {
      rootFlag += 0x00007000;       // Class has inherits from TObject
    }                               //
    if ( fClass.TypeInfo() == typeid(std::string) )  {
      rootFlag = 0x48F00;
    }

    fSetup_memvar  = Allocate_void_function(this, Setup_memvar_with_context );
    fSetup_memfunc = Allocate_void_function(this, Setup_memfunc_with_context );

    G__tagtable_setup( fTaginfo->tagnum,    // tag number
                       fClass.SizeOf(),     // size
                       G__CPPLINK,           // cpplink
                       rootFlag,             // isabstract
                       comment.empty() ? 0 : comment.c_str(), // comment
                       fSetup_memvar,       // G__setup_memvarMyClass
                       fSetup_memfunc);     // G__setup_memfuncMyClass
  }

  void CINTClassBuilder::Setup_memfunc_with_context(void* ctx) {
    ((CINTClassBuilder*)ctx)->Setup_memfunc();
  }
  void CINTClassBuilder::Setup_memvar_with_context(void* ctx) {
    ((CINTClassBuilder*)ctx)->Setup_memvar();
  }

  void CINTClassBuilder::Setup_memfunc() {

    for ( size_t i = 0; i < fClass.FunctionMemberSize(); i++ ) 
      CINTScopeBuilder::Setup(fClass.FunctionMemberAt(i).TypeOf());

    G__tag_memfunc_setup(fTaginfo->tagnum);
    for ( size_t i = 0; i < fClass.FunctionMemberSize(); i++ ) {
      Member method = fClass.FunctionMemberAt(i); 
      std::string n = method.Name();
      CINTFunctionBuilder::Setup(method);
    }
    ::G__tag_memfunc_reset();
  }

  void CINTClassBuilder::Setup_memvar() {

    for ( size_t i = 0; i < fClass.DataMemberSize(); i++ ) 
      CINTScopeBuilder::Setup(fClass.DataMemberAt(i).TypeOf());

    const char* ref_t = "pool::Reference";
    const char* tok_t = "pool::Token";
    G__tag_memvar_setup(fTaginfo->tagnum);
    // Set placeholder for virtual function table if the class is virtual
    if ( fClass.IsVirtual() ) {
      G__memvar_setup((void*)0,'l',0,0,-1,-1,-1,4,"G__virtualinfo=",0,0);
    }

    if ( ! IsSTL(fClass.Name(SCOPED)) )  {
      for ( size_t i = 0; i < fClass.DataMemberSize(); i++ ) {
        Member dm = fClass.DataMemberAt(i);
        char* comment = NULL;
        std::string cm = dm.Properties().HasKey("comment") ? 
          dm.Properties().PropertyAsString("comment") : std::string("");

        Type t = dm.TypeOf();
        while ( t.IsTypedef() ) t = t.ToType();
        if ( !t && dm.IsTransient() )  {
          if( Cintex::Debug() ) std::cout << "Ignore transient MemberNth: " << fName << "::" 
            << dm.Name() << " [No valid reflection class]" << std::endl;
          continue;
        }
        else if ( !t )  {
          if( Cintex::Debug() > 0 )  {
            std::cout << "WARNING: Member: " << fName << "::" 
                      << dm.Name() << " [No valid reflection class]"
                      << std::endl;
          }
          //throw std::runtime_error("Member: "+fName+"::"+dm.Name()+" [No valid reflection class]");
        }
        //--- Add the necessary artificial comments ------ 
        if ( dm.IsTransient() ||
             IsSTL(fName) || 
             IsTypeOf(fClass, ref_t) || 
             IsTypeOf(fClass, tok_t) )  {
          char* com = new char[cm.length()+4];
          ::sprintf(com,"! %s",cm.c_str());
          comment = com;
          CommentBuffer::Instance().add(comment);
        }
        else if ( (t.IsClass() || t.IsStruct()) && 
                  (IsTypeOf(t,ref_t) || IsTypeOf(t,tok_t)) )  {
          char* com = new char[cm.length()+4];
          ::sprintf(com,"|| %s",cm.c_str());
          comment = com;
          CommentBuffer::Instance().add(comment);
        }
        else if ( !cm.empty() )  {
          char* com = new char[cm.length()+4];
          ::strcpy(com, cm.c_str());
          comment = com;
          CommentBuffer::Instance().add(comment);
        }
        Indirection  indir = IndirectionGet(dm.TypeOf());
        CintTypeDesc type = CintType(indir.second);
        string dname = dm.Properties().HasKey("ioname") ? 
                       dm.Properties().PropertyAsString("ioname") : dm.Name();
        ostringstream ost;
        if ( t.IsArray() ) ost << dname << "[" << t.ArrayLength() << "]=";
        else               ost << dname << "=";
        string expr = ost.str();
        int member_type     = type.first;
        int member_indir    = 0;
        int member_tagnum   = -1;
        int member_typnum   = -1;
        int member_isstatic = dm.IsStatic() ? G__LOCALSTATIC : G__AUTO;
        switch(indir.first)  {
          case 0: 
            break;
          case 1:
            member_type -= 'a'-'A';            // if pointer: 'f' -> 'F' etc.
            break;
          default:
            member_type -= 'a'-'A';            // if pointer: 'f' -> 'F' etc.
            member_indir = indir.first;
          break;
        }

        if ( type.first == 'u' )  {
          //dependencies.push_back(indir.second);
          member_tagnum = CintTag(type.second);
          if ( typeid(longlong) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__LONGLONG);
          else if ( typeid(ulonglong) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__ULONGLONG);
          else if ( typeid(long double) == indir.second.TypeInfo() )
            ::G__loadlonglong(&member_tagnum, &member_typnum, G__LONGDOUBLE);
        }

        int member_access = 0;
        if ( dm.IsPrivate() )        member_access = G__PRIVATE;
        else if ( dm.IsProtected() ) member_access = G__PROTECTED;
        else if ( dm.IsPublic() )    member_access = G__PUBLIC;

        if ( Cintex::Debug() > 2 )  {
          std::cout
          << std::setw(16) << std::left << "declareField>"
          << "  [" << char(member_type) 
          << "," << std::right << std::setw(3) << dm.Offset()
          << "," << std::right << std::setw(2) << member_indir 
          << "," << std::right << std::setw(3) << member_tagnum
          << "] " 
          << (dm.TypeOf().IsConst() ? "const " : "")
          << std::left << std::setw(7)
          << (G__AUTO==member_isstatic ? "auto " : "static ")
          << std::left << std::setw(24) << dname
          << " \"" << (char*)(comment ? comment : "(None)") << "\""
          << std::endl
          << std::setw(16) << std::left << "declareField>"
          << "  Type:" 
          << std::left << std::setw(24) << "["+t.Name(SCOPED)+"]"
          << " DeclBy:" << fTaginfo->tagname
          << std::endl;
        }
        ::G__memvar_setup((void*)dm.Offset(),                         // p
                          member_type,                                // type
                          member_indir,                               // indirection
                          dm.TypeOf().IsConst(),                        // const
                          member_tagnum,                              // tagnum
                          member_typnum,                              // typenum
                          member_isstatic,                            // statictype
                          member_access,                              // accessin
                          expr.c_str(),                               // expression
                          0,                                          // define macro
                          comment                                     // comment
        );
      }
    }
    G__tag_memvar_reset();
  }

  CINTClassBuilder::Bases* CINTClassBuilder::GetBases() {
    if ( fBases ) return fBases;
    Member getbases = fClass.MemberByName("__getBasesTable");
    if( getbases ) {
      fBases = (Bases*)( getbases.Invoke().Address() );
    }
    else {
      static Bases s_bases;
      fBases = &s_bases;
    }
    return fBases;
  }

  void CINTClassBuilder::Setup_inheritance_simple() {
    if ( 0 == ::G__getnumbaseclass(fTaginfo->tagnum) )  {     
      bool IsVirtual = false; 
      for ( Bases::iterator it = GetBases()->begin(); it != GetBases()->end(); it++ )
         if( (*it).first.IsVirtual() ) IsVirtual = true;

      if ( IsVirtual ) {
        if ( !fClass.IsAbstract() )  {
          Member ctor, dtor;
          for ( size_t i = 0; i < fClass.FunctionMemberSize(); i++ ) {
            Member method = fClass.FunctionMemberAt(i); 
            if( method.IsConstructor() && method.FunctionParameterSize() == 0 )  ctor = method;
            else if ( method.IsDestructor() )  dtor = method;
          }
          if ( ctor )  {
            Object obj = fClass.Construct();
            Setup_inheritance_simple(obj);
            if ( dtor ) fClass.Destruct(obj.Address());
          }
          // There is no default constructor. So, it is not a I/O class
          else {  
            Object obj(fClass, 0);
            Setup_inheritance_simple(obj);
          }
        }
        // Special case of "pure abstract". The offsets will be Set to 0.
        // All that is necessary because ROOT does not handle virtual inheritance correctly.
        // ROOT always wants to Get a real offset between base classes ans this is not 
        // possible without having an Instance of the object. In case of pure abstract classes
        // we can not do it. So, in case the abstract class has no data members then we assume
        // offsets to base class to be 0.
        else if ( fClass.IsAbstract() && fClass.DataMemberSize() == 0) {
          Object obj(fClass, 0);
          Setup_inheritance_simple(obj);
        }
        // The above fails for Gaudi Algorithms (virutal inheritance, abstract and with
        // data members. Do not know What to do.
        else {  
          Object obj(fClass, 0);
          Setup_inheritance_simple(obj);
        }
      }
      else {
        Object obj(fClass, (void*)0x100);
        Setup_inheritance_simple(obj);
      }
    }
  }
  void CINTClassBuilder::Setup_inheritance_simple(Object& obj) {
    if ( ! IsSTL(fClass.Name(SCOPED)) )    {
      if ( 0 == ::G__getnumbaseclass(fTaginfo->tagnum) )  {
        for ( Bases::iterator it = GetBases()->begin(); it != GetBases()->end(); it++ ) {
          Base base  = it->first;
          int  level = it->second;
          Type btype = base.ToType();
          CINTScopeBuilder::Setup(btype);
          std::string b_nam = CintName(btype);
          int b_tagnum = CintTag(b_nam);
          // Get the Offset. Treat differently virtual and non-virtual inheritance
          size_t offset;
          long  type = (level == 0) ?  G__ISDIRECTINHERIT : 0;
          if ( base.IsVirtual() ) {
            if (obj.Address())  {
              offset = ( * base.OffsetFP())(obj.Address());
            }
            else {
              offset = (size_t)base.OffsetFP();
              type = type | G__ISVIRTUALBASE;
            }
          }
          else {
            offset = base.Offset((void*)0x100);
          }
          if( Cintex::Debug() > 1 )  {
            std::cout << fClass.Name(SCOPED) << " Base:" << btype.Name(SCOPED) << " Offset:" << offset << std::endl;
          }
          int mod = base.IsPublic() ? G__PUBLIC : ( base.IsPrivate() ? G__PRIVATE : G__PROTECTED );
          ::G__inheritance_setup(fTaginfo->tagnum, b_tagnum, offset, mod, type );
          Object bobj(btype,(char*)obj.Address() + offset);
          //CINTClassBuilder::Get(btype).Setup_inheritance_simple(bobj);
        }
      }
    }    
  }
  
  void CINTClassBuilder::Setup_inheritance() {
    Member GetBases = fClass.MemberByName("__getBasesTable");
    if( GetBases ) {
      typedef vector<pair<Base,int> > Bases;
      Bases* bases = (Bases*)(GetBases.Invoke().Address());
      for ( Bases::iterator it = bases->begin(); it != bases->end(); it++ ) {
        Base base  = it->first;
        int  level = it->second;
        int b_tagnum = CintTag(base.ToType().Name(SCOPED));
        // Get the offset. Treat differently virtual and non-virtual inheritance
        size_t offset;
        long  type = level == 0 ?  G__ISDIRECTINHERIT : 0;
        if ( base.IsVirtual() ) {
          offset = (size_t) base.OffsetFP();
          type = type | G__ISVIRTUALBASE;
        }
        else {
          offset = base.Offset((void*)0x100);
        }
        int mod = base.IsPublic() ? G__PUBLIC : ( base.IsPrivate() ? G__PRIVATE : G__PROTECTED );
        G__inheritance_setup(fTaginfo->tagnum, b_tagnum, offset, mod, type );
      }    
    }
    else {
      Setup_inheritance( fTaginfo->tagnum, 0, fClass, G__ISDIRECTINHERIT);
      // check if pending bases are needed to be resolved
      for ( list<PendingBase>::iterator it = pendingBases().begin(); it != pendingBases().end();) {
        if ( (*it).basetype == fClass ) {
          list<PendingBase>::iterator curr = it++;
          Setup_inheritance((*curr).tagnum, (*curr).offset, (*curr).basetype, 0);
          pendingBases().erase(curr);
        }
        else {
          it++;
        }
      }
    }
  }  
  
  void CINTClassBuilder::Setup_inheritance(int tagnum, size_t /* off */, const Type& cl, int ind) {
    for ( size_t i = 0; i < cl.BaseSize(); i++ ) {
      Base base = cl.BaseAt(i);
      int b_tagnum = CintTag(base.ToType().Name(SCOPED));
      // Get the offset. Treat differently virtual and non-virtual inheritance
      size_t offset;
      long  type = ind;
      if ( base.IsVirtual() ) {
        offset = (size_t) base.OffsetFP();
        type = type | G__ISVIRTUALBASE;
      }
      else {
        offset = base.Offset((void*)0x100);
      }
      int mod = base.IsPublic() ? G__PUBLIC : ( base.IsPrivate() ? G__PRIVATE : G__PROTECTED );
      G__inheritance_setup(tagnum, b_tagnum, offset, mod, type );
      // scan next level of base classes recursively if already loaded otherwise add into 
      // the pending list
      if( base.ToType() ) Setup_inheritance(tagnum, offset, base.ToType(), 0);
      else                pendingBases().push_back(PendingBase(base.ToType(),tagnum, offset));
    }
  }

  void CINTClassBuilder::Setup_typetable() {
  }

}}
