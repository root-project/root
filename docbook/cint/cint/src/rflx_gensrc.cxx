/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//$Id: rflx_gensrc.cxx,v 1.13 2006/11/24 13:14:37 rdm Exp $

#include "rflx_gensrc.h"
#include "rflx_tools.h"

#include "G__ci.h"
#include "common.h"
#include "global.h"
#include "Api.h"

#include <ctime>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <list>

// Meaning of long Property() bits
// 
//  9 public
// 10 protected
// 11 private
// 21 virtual

// ______________________________________________________________________________
/* Reflex dictionary source code can be divided into different parts. 
*
* 1. The "header" contains general includes and defines (mostly static)
* 2. The "type dictionaries" will generate a 'stub' for every C++ type contained
*    in the file (e.g. every base class type, parameter type, typedef, etc)
* 3. The "class dictionaries" contain the class descriptions (including stub functions)
* 4. The "instances" will instantiate the dictionaries with a static invocation once loaded
*
* These different parts are generated in the file below and then put on a filestream
* together in the dictionary file 
*/

// Fix for C2039 SDK61 bug
#if defined(MSVCVER9) && defined(WSDK61)
namespace std {
 // TEMPLATE FUNCTION _Swap_adl
 template<class _Ty> inline void _Swap_adl(_Ty& _Left, _Ty& _Right) {	// exchange values stored at _Left and _Right, using ADL
  swap(_Left, _Right);
 }
}
#endif

void rflx_gensrc::gen_file()
{

   gen_header();

   gen_classdicts();

   gen_dictinstances();

   gen_freefundicts();

   gen_freevardicts();

   gen_typedefdicts();

   gen_enumdicts();

   gen_typedicts();

   std::ofstream s(m_dictfile.c_str());
   s << m_hd.str()
     << m_td.str();

   m_shadowMaker.WriteAllShadowClasses();
   s << m_sh.str()
     << m_cd.str()
     << m_ff.str()
     << m_fv.str()
     << m_di.str();
   s.close();

}

// ______________________________________________________________________________
/* Every type which appears in the dictionary will be generated as a 'stub'. Once
* the type (e.g. class) gets fully defined, the dictionary information will be 
* added to this stub information. Otherwise what is generated here will be the 
* only information available. 'Stubs' are generated for every type appearing in
* the dictionary source code (e.g. function parameter/return types, data members, ...) 
*/
std::string rflx_gensrc::gen_type(G__TypeInfo & tn)
{
   // Q: volatile supported?
   // reference, pointer, const, volatile, type, typedef

   std::string tName = tn.Name();
   if (tName == "(unknown)") {
      // try class
      G__ClassInfo ci(tn.Tagnum());
      if (ci.IsValid()) 
         return gen_type(ci);
   }

   std::ostringstream tvNumS("");
   tvNumS << m_typeNum;
   std::string tvNumStr = "type_" + tvNumS.str();

   TypeMap::const_iterator mIt = m_typeMap.find(tName);
   if (mIt != m_typeMap.end())
      return m_typeMap[tName];
   else
      m_typeMap[tName] = tvNumStr;
   ++m_typeNum;

   if (tn.Name()[strlen(tn.Name()) - 1] == '*') {
      G__TypeInfo ti(tName.substr(0, tName.rfind("*")).c_str());
      m_typeVec.push_back("Type " + tvNumStr + " = PointerBuilder(" +
                          gen_type(ti) + ");");
   } else if (tn.Property() & G__BIT_ISTYPEDEF && tn.Type() != 'u') {
      G__TypedefInfo tdi(tn.Typenum());
      if (!tdi.IsValid())
         m_typeVec.push_back("Type " + tvNumStr + " = TypeBuilder(\"" +
                             tName + "\");");
      else {
         G__TypeInfo clOrig(tdi.TrueName());
         m_typeVec.push_back("Type " + tvNumStr + " = TypedefTypeBuilder(\"" + tName 
            + "\", " + gen_type(clOrig) + ");");
      }
   } else if (tn.Reftype()) {
      G__TypeInfo ti(tName.substr(0, tName.rfind("&")).c_str());
      m_typeVec.push_back("Type " + tvNumStr + " = ReferenceBuilder(" +
                          gen_type(ti) + ");");
   } else if (tn.Isconst()&G__CONSTVAR) {
      std::string::size_type posConst=tName.rfind("const");
      if (posConst != std::string::npos) {
         if (posConst)
            tName = tName.substr(0, tName.length() - 5);
         else
            tName = tName.substr(6);
         G__TypeInfo ti(tName.c_str());
         m_typeVec.push_back("Type " + tvNumStr + " = ConstBuilder(" +
                             gen_type(ti) + ");");
      } else 
         m_typeVec.push_back("Type " + tvNumStr + " = TypeBuilder(\"" +
                             tName + "\");");
   } else {
      m_typeVec.push_back("Type " + tvNumStr + " = TypeBuilder(\"" +
                          tName + "\");");
   }
   return tvNumStr;
}


// ______________________________________________________________________________
/* Every type which appears in the dictionary will be generated as a 'stub'. Once
* the type (e.g. class) gets fully defined, the dictionary information will be 
* added to this stub information. Otherwise what is generated here will be the 
* only information available. 'Stubs' are generated for every type appearing in
* the dictionary source code (e.g. function parameter/return types, data members, ...) 
*/
std::string rflx_gensrc::gen_type(G__ClassInfo & tn)
{
   // Q: volatile supported?
   // reference, pointer, const, volatile, type, typedef

   //std::string tName = rflx_tools::decorate_stl_type(tn.Name());
   std::string tName = tn.Fullname();
   std::ostringstream tvNumS("");
   tvNumS << m_typeNum;
   std::string tvNumStr = "type_" + tvNumS.str();

   TypeMap::const_iterator mIt = m_typeMap.find(tName);
   if (mIt != m_typeMap.end())
      return m_typeMap[tName];
   else
      m_typeMap[tName] = tvNumStr;
   ++m_typeNum;

   m_typeVec.push_back("Type " + tvNumStr + " = TypeBuilder(\"" +
                       tName + "\");");

   return tvNumStr;
}


// ______________________________________________________________________________
/* This function generates the (mostly) static information which will appear in 
* the header of the dictionary source code.
*/
void rflx_gensrc::gen_header()
{
   time_t t;
   time(&t);
   m_hd <<
       "// Do not modify this file. Generated automatically by rootcint on "
       << ctime(&t)
       << "#ifdef _WIN32" << std::
       endl << "#pragma warning ( disable : 4786 )" << std::
       endl << "#ifndef LCGDICT_STRING" << std::
       endl <<
       "#include <string> // Included here since it is sensitive to private->public trick"
       << std::endl << "#endif" << std::endl << "#endif" << std::
       endl << "#define private public" << std::
       endl << "#define protected public" << std::
       endl << "#include \"" << m_sourcefile << "\"" << std::
       endl << "#undef private" << std::endl << "#undef protected" << std::
       endl << "#include \"Reflex/Builder/ReflexBuilder.h\"" << std::
       endl << "#include \"Cintex/Cintex.h\"" << std::
       endl << "#include <typeinfo>" << std::
       endl << "namespace ROOT { namespace Reflex { } }" << std::
       endl <<
       "namespace seal { namespace reflex { using namespace ROOT::Reflex; } }"
       << std::endl << "using namespace seal::reflex;" << std::
       endl << "using namespace std;" << std::endl << std::endl;
}


// ______________________________________________________________________________
/* This function will generated the dictionary stub information for types. 
*/
void rflx_gensrc::gen_typedicts()
{
   ind.clear();
   m_td << "//" << std::endl;
   m_td << "// ---------- Dictionary type generation ----------" << std::
       endl;
   m_td << "//" << std::endl;
   m_td << "namespace {" << std::endl;
   ++ind;
   m_td << ind() << "Type type_void = TypeBuilder(\"void\");" << std::endl;
   for (std::vector < std::string >::const_iterator it = m_typeVec.begin();
        it != m_typeVec.end(); ++it) {
      m_td << ind() << *it << std::endl;
   }
   --ind;
   m_td << "}" << std::endl << std::endl;
}

// ______________________________________________________________________________
/* Loops over all classes defined and generates dictionary information for it which
* is put on a stream
*/
void rflx_gensrc::gen_classdicts()
{
   G__ClassInfo ci;

   // loop over classes
   while (ci.Next()) {

      char type = G__struct.type[ci.Tagnum()];
      // if pragma link C++ class is set and the type is "class/struct"
      if (ci.Linkage() == G__CPPLINK && (type == 'c' || type == 's')
          && ci.IsLoaded()) {

         //m_classNames.push_back(rflx_tools::decorate_stl_type(ci.Fullname()));
         std::string fclname=ci.Fullname();
         if (!strncmp(fclname.c_str(),"pair",4))
            fclname.insert(0,"std::");

         m_classNames.push_back(fclname);

         gen_type(ci);
         gen_classdictdefs(ci);
         if (m_split)
            gen_classdictdecls(m_cds, ci);
         else
            gen_classdictdecls(m_cd, ci);
      }
   }
}


// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_typedefdicts()
{
   G__TypedefInfo td;

   while (td.Next()) {

      if (G__newtype.globalcomp[td.Typenum()]) {

         // G__type2string(td.Type(),td.Tagnum(),td.Typenum(),td.Reftype(),td.Isconst())

         std::string tName = td.TrueName();
         if (tName.length()) {
            std::string tdName = td.Name();
            if (m_typeMap.find(tdName) == m_typeMap.end()) {
               std::ostringstream tvNumS("");
               tvNumS << m_typeNum;
               std::string tvNumStr = "type_" + tvNumS.str();
               m_typeMap[tdName] = tvNumStr;
               ++m_typeNum;
               G__TypeInfo ti(tName.c_str());
               m_typeVec.push_back("Type " + tvNumStr +
                                   " = TypedefTypeBuilder(\"" + tdName +
                                   "\", " + gen_type(ti) + ");");
            }
         }
      }
   }
}



// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_enumdicts()
{
   ind.clear();
   std::string cldname =
       "__reflex__enums__dict__" +
       rflx_tools::escape_class_name(m_sourcefile);
   m_fv << ind() << "//" << std::endl;
   m_fv << ind() <<
       "// ---------- Dictionary for enums ----------" << std::
       endl;
   m_fv << ind() << "//" << std::endl;
   m_fv << ind() << "class " << cldname << " {" << std::endl;
   m_fv << ind() << "public:" << std::endl;
   ++ind;
   m_fv << ind() << cldname << "();" << std::endl;
   --ind;
   m_fv << ind() << "};" << std::endl << std::endl;
   m_fv << ind() << cldname << "::" << cldname << "() {" << std::endl;
   ++ind;

   G__ClassInfo en;
   while (en.Next()) {
      if (!(en.Property() & G__BIT_ISENUM) 
         || en.Linkage() != G__CPPLINK 
         || en.Name()[0] == '$')  continue;

      std::string fqiParent;
      G__ClassInfo parent = en.EnclosingClass();
      while (parent.IsValid()) {
         fqiParent.insert(0,"::");
         fqiParent.insert(0,parent.Name());
         parent = parent.EnclosingClass();
      }
      m_fv << ind() << "EnumBuilder(\"" << fqiParent << en.Name() 
           << "\", typeid(" << fqiParent << en.Name() << "))";

      G__ClassInfo enc=en.EnclosingClass();
      G__DataMemberInfo dmEnc(enc);
      G__DataMemberInfo dmGlob;
      G__DataMemberInfo *dm = &dmGlob;
      if (enc.IsValid()) dm = &dmEnc;
      while (dm->Next())
         if (dm->Type() && dm->Type()->Tagnum() == en.Tagnum() 
            && (dm->Property() & (G__BIT_ISENUM | G__BIT_ISSTATIC | G__BIT_ISCONSTANT))) {
            m_fv << std::endl << ind() << ".AddItem(\"" << dm->Name() << "\" , "
               << fqiParent << dm->Name() << ")";
         }
      m_fv << ";" << std::endl;
   }
   --ind;
   m_fv << ind() << "};" << std::endl << std::endl;
}


// ______________________________________________________________________________
/* A function to generate the declaration of a stub function for structors and
* general methods (appears inside the class body of the dictionary class)
*/
void rflx_gensrc::gen_decl(char type, int num, const std::string & clname,
                           const std::string & fclname)
{
   std::ostringstream o;
   m_cd << ind() << "static void * ";
   switch (type) {
   case 'm':
      m_cd << "method_" << num << "(void*,";
      break;
   case 'c':
      m_cd << "constructor_" << num << "(void*,";
      break;
   case 'd':
      m_cd << "destructor(void* o,";
      break;
   default:
      break;
   }
   m_cd << " const std::vector<void*>&, void*)";
   if (type == 'd')
      m_cd << " {" << std::endl
          << ind() << "  ((::" << fclname << "*)o)->~" << clname << "();"
          << std::endl << ind() << "  return 0;" << std::
          endl << ind() << "}" << std::endl;
   else
      m_cd << ";" << std::endl;
}


// ______________________________________________________________________________
/* Generates one dictionary class declaration, i.e. the declarations for the stub
* functions and the declaration for the constructor of the dictionary class plus
* the definition of the dictionary class constructor (defining the meta information
* for base classes, data/function members)
*/
void rflx_gensrc::gen_classdictdefs(G__ClassInfo & ci)
{
   // Q: distinction class/struct?
   // Q: compiler auto generated structors?
   std::string cl_modifiers = "CLASS";
   std::string clname = ci.Name();
   //std::string fclname = rflx_tools::decorate_stl_type(ci.Fullname());
   std::string fclname = ci.Fullname();

   if (!strncmp(fclname.c_str(),"pair",4))
      fclname.insert(0,"std::");

   std::string cldname =
       "__" + rflx_tools::escape_class_name(fclname) + "_dict";
   ind.clear();
   m_cd << ind() << "//" << std::endl;
   m_cd << ind() << "// ---------- Dictionary for class " << fclname <<
       " ----------" << std::endl;
   m_cd << ind() << "//" << std::endl;
   m_cd << ind() << "class " << cldname << " {" << std::endl;
   m_cd << ind() << "public:" << std::endl;
   ++ind;
   m_cd << ind() << cldname << "();" << std::endl;
   int mNum = -1;
   int cNum = -1;

   bool hasConstructor = false;
   G__MethodInfo fm(ci);
   while (fm.Next()) {
      if (strlen(fm.Name())) {
         std::string fmname = fm.Name();
         if (fmname == clname)
            hasConstructor = true;
         if (!(fm.Property() & G__BIT_ISPUBLIC))
            continue;

         if (fmname == clname) {
            gen_decl('c', ++cNum);
         }                      // constructor
         else if (fmname[0] == '~') {
            gen_decl('d', 0, clname, fclname);
         }                      // destructor
         else {
            gen_decl('m', ++mNum);
         }                      // regular function
      }
   }
   if (!hasConstructor) {
      if (ci.Property() & G__BIT_ISABSTRACT) {
         m_cd << ind() <<
          "static void * constructor_auto(void*, const std::vector<void*>&, void*) { ";
         m_cd << " return 0; }" << std::endl;
      } else {
         m_cd << ind() <<
          "static void * constructor_auto(void* mem, const std::vector<void*>&, void*) { ";
         m_cd << "return ::new(mem) ::" << fclname << "(); }" << std::endl;
      }
   }
   // generate method __getBasesTable returning base classes
   m_cd << ind() << "static void* " << cldname 
      << "::method_getBaseTable( void*, const std::vector<void*>&, void*);" << std::endl;
   --ind;

   m_cd << ind() << "};" << std::endl << std::endl;
   m_cd << ind() << cldname << "::" << cldname << "() {" << std::endl;
   ++ind;
   m_cd << ind() << "ClassBuilder(\"" << fclname << "\", typeid(" << fclname 
      << "), sizeof(" << fclname <<"), " << cl_modifiers << ")"; 

   gen_baseclassdefs(ci);
   gen_datamemberdefs(ci);
   gen_functionmemberdefs(ci);

   --ind;
   m_cd << ";" << std::endl << ind() << "}" << std::endl << std::endl;

}


// ______________________________________________________________________________
/* Generate the meta information for the base classes of a class
*/
void rflx_gensrc::gen_baseclassdefs(G__ClassInfo & ci)
{
   // Q: more bits set in Property(), what do they mean? e.g. 0, 17
   G__BaseClassInfo bc(ci);
   while (bc.Next()) {
      long bcProp = bc.Property();
      if (  (bcProp & G__BIT_ISVIRTUALBASE) &&
           !(bcProp & G__BIT_ISDIRECTINHERIT)) {
         // CINT duplicates the remote virtual base class in the list scanned
         // by G__BaseClassInfo, we need to skip them.
         continue;
      }

      G__TypeInfo bct(bc.Name());
      std::string dict_typealias=gen_type(bct);
      std::string bc_modifiers;
      if (bcProp & (1 << 9))
         bc_modifiers += "PUBLIC";
      else if (bcProp & (1 << 10))
         bc_modifiers += "PROTECTED";
      else if (bcProp & (1 << 11))
         bc_modifiers += "PRIVATE";
      if (bcProp & (1 << 21))
         bc_modifiers += " | VIRTUAL";
      std::string bcName(bc.Fullname()); // Fullname uses static buf, cannot call on ci and bc
      m_cd << std::endl
           << ind() << ".AddBase(" << dict_typealias << ", BaseOffset< " 
           << ci.Fullname() << ", " << bcName << " >::Get(), " << bc_modifiers << ")";

   }
}


// ______________________________________________________________________________
/* Generate the meta information for the data members of a class
*/
void rflx_gensrc::gen_datamemberdefs(G__ClassInfo & ci)
{
   G__DataMemberInfo dm(ci);
   while (dm.Next()) {
      if (strcmp("G__virtualinfo", dm.Name()) != 0
          && !(dm.Property() & G__BIT_ISSTATIC)) {
         std::string dm_modifiers;
         long dmProp = dm.Property();
         if (dmProp & (1 << 9))
            dm_modifiers += "PUBLIC";
         else if (dmProp & (1 << 10))
            dm_modifiers += "PROTECTED";
         else if (dmProp & (1 << 11))
            dm_modifiers += "PRIVATE";
         /*
            int index = dm.Index();
            G__var_array * var = (G__var_array*)dm.Handle();
            switch (var->access[index]) {
            case G__PUBLIC:    dm_modifiers = "PUBLIC";    break;
            case G__PROTECTED: dm_modifiers = "PROTECTED"; break;
            case G__PRIVATE:   dm_modifiers = "PRIVATE";   break;
            default:                                       break;
            }
            if (var->statictype[index] != -1) dm_modifiers += " | STATIC";
          */
         //std::string fclname = rflx_tools::decorate_stl_type(ci.Fullname());

         // don't mangle nested types
         G__ClassInfo ciUpper=ci;
         int level=0;
         while (ciUpper.EnclosingClass().IsValid() 
            && (ciUpper.EnclosingClass().Property() & G__BIT_ISCLASS)) {
            ciUpper=ciUpper.EnclosingClass();
            ++level;
         }

         std::string fclname;
         m_shadowMaker.GetFullShadowName(ci, fclname);

         size_t pos = 0;
         int i = 0;
         while ((pos = fclname.find(",", pos + 1)) != std::string::npos)
            ++i;
         std::string offnum = "";
         if (i) {
            std::stringstream s;
            s << ++i;
            offnum = s.str();
         }
         m_cd << std::endl
              << ind() << ".AddDataMember(" << gen_type(*dm.Type()) << ", \"" << dm.Name() << "\", ";
         // no (real) shadow means no offsets
         if (m_shadowMaker.NeedShadowCached(ci.Tagnum())!=1) 
            m_cd << "0, ";
         else
            m_cd << "OffsetOf" << offnum << "(" << fclname << ", " << dm.Name() << "), ";
         m_cd << dm_modifiers << ")";
         // comment
         G__FastAllocString buf(16*1024);
         buf[0] = 0;
         G__var_array* va = (G__var_array*)dm.Handle();
         G__getcomment(buf,&va->comment[dm.Index()],va->tagnum);
         if (buf[0]) {
            std::string comment(buf);
            std::string::size_type posRepl = 0;
            // replace '"' with '\"' and '\' with '\\'
            while (std::string::npos != (posRepl = comment.find_first_of("\\\"", posRepl))) {
               comment.insert(posRepl, "\\");
               posRepl+=2; // skip inserted '\' and found '"'
            }
            m_cd << std::endl
                 << ind() << ".AddProperty(\"comment\",\"" << comment << "\")";
         }
      }
   }
}


// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_parTypesNames(std::string & retParTypes,
                                    std::string & parNames,
                                    G__MethodInfo & fm)
{
   G__MethodArgInfo ma(fm);
   bool subseq = false;
   while (ma.Next()) {
      if (subseq)
         parNames += ";";
      const char *maName = ma.Name();
      if (maName) {
         parNames += std::string(ma.Name());
         if (ma.DefaultValue()) {
            std::string maDef = ma.DefaultValue();
            size_t posEsc = 0;
            while (std::string::npos != (posEsc = maDef.find('"', posEsc))) {
               maDef.replace(posEsc, 1, "\\\"");
               posEsc += 2;
            }
            parNames += "=" + maDef;
         }
      }
      retParTypes += "," + gen_type(*ma.Type());
      subseq = true;
   }
}


// ______________________________________________________________________________
/* Generate the meta information for the function members of a class
*/
void rflx_gensrc::gen_functionmemberdefs(G__ClassInfo & ci)
{
   // Q: automatic generation con/de/copycon-structor?
   // Q: virtual seems not to work
   int mNum = -1;
   int cNum = -1;
   std::string clname = "";
   if (strlen(ci.Name()))
      clname = ci.Name();
   G__MethodInfo fm(ci);
   bool hasConstructor = false;
   while (fm.Next()) {
      std::string fmname = fm.Name();

      if (fmname.length()) {
         std::string fm_modifiers = "";

         G__ifunc_table_internal *ift = G__get_ifunc_internal((G__ifunc_table *) fm.Handle());
         int index = fm.Index();

         switch (ift->access[index]) {
         case G__PUBLIC:
            fm_modifiers = "PUBLIC";
            break;
         case G__PROTECTED:
            fm_modifiers = "PROTECTED";
            break;
         case G__PRIVATE:
            fm_modifiers = "PRIVATE";
            break;
         default:
            break;
         }

         //if ((ift->isvirtual[index] != -1) || (ift->ispurevirtual[index] != -1)) fm_modifiers += " | VIRTUAL";

         std::string retParTypes = "";
         std::string parNames = "";

         bool isConstructor = false;
         if (fmname == clname) {
            isConstructor = true;
            hasConstructor = true;
         }

         if (!(fm.Property() & G__BIT_ISPUBLIC))
            continue;

         bool isDestructor = false;
         if (fmname[0] == '~')
            isDestructor = true;

         if (isConstructor || isDestructor)
            retParTypes += "type_void";
         else
            retParTypes += gen_type(*fm.Type());

         gen_parTypesNames(retParTypes, parNames, fm);

         if (isConstructor) {
            fm_modifiers += " | CONSTRUCTOR";
            m_cd << std::endl
                << ind() << ".AddFunctionMember(FunctionTypeBuilder(" <<
                retParTypes << "), \"" << clname << "\", constructor_" <<
                ++cNum << ", 0, \"" << parNames << "\", " << fm_modifiers
                << ")";
         } else if (isDestructor) {
            fm_modifiers += " | DESTRUCTOR";
            m_cd << std::endl
                << ind() << ".AddFunctionMember(FunctionTypeBuilder(" <<
                retParTypes << "), \"~" << clname << "\", destructor" <<
                ", 0, \"" << parNames << "\", " << fm_modifiers << ")";
         }
         // regular function
         else {
            m_cd << std::endl
                << ind() << ".AddFunctionMember(FunctionTypeBuilder(" <<
                retParTypes << "), \"" << fmname << "\", method_" << ++mNum
                << ", 0, \"" << parNames << "\", " << fm_modifiers << ")";
         }
      }
   }
   if (!hasConstructor) {

      // this is a huge overhead in order to find the proper type for the pointer to class
      std::string fullclname = ci.Fullname();
      std::string cltypestr;
      if (m_typeMap.find(fullclname + "*") != m_typeMap.end())
         cltypestr = m_typeMap[fullclname + "*"];
      else if (m_typeMap.find(fullclname) != m_typeMap.end()) {
         std::ostringstream tvNumS("");
         tvNumS << m_typeNum;
         cltypestr = "type_" + tvNumS.str();
         ++m_typeNum;
         m_typeVec.push_back("Type " + cltypestr + " = PointerBuilder(" +
                             m_typeMap[fullclname] + ");");
      } else {
         std::
             cerr << "makecint: could not find type information for type "
             << fullclname << std::endl;
      }

      m_cd << std::endl
          << ind() << ".AddFunctionMember(FunctionTypeBuilder(" <<
          cltypestr << "), \"" << clname <<
          "\", constructor_auto, 0, \"\", PUBLIC | CONSTRUCTOR)";
   }

   // Add method __getBasesTable returning base classes
   m_cd << std::endl << ind() 
        << ".AddFunctionMember<void*(void)>(\"__getBasesTable\", "
        << "method_getBaseTable, 0, 0, PUBLIC | ARTIFICIAL)";
}


// ______________________________________________________________________________
/*
*
*/
int rflx_gensrc::gen_stubfuncdecl_header(std::ostringstream & s,
                                         G__MethodInfo & fm,
                                         const std::string & objcaststr,
                                         int argNum)
{
   if (argNum < 0)
      argNum = 0;
   int moffset = 0;
   std::string fmname = fm.Name();
   std::string retname = rflx_tools::rm_end_ref(fm.Type()->Name());
   int index = fm.Index();
   G__ifunc_table_internal *var = G__get_ifunc_internal((G__ifunc_table *) fm.Handle());
   G__SIGNEDCHAR_T retT = var->type[index];
   // pointer
   if (isupper(retT)) {
      s << ind() << "return (void*)" << objcaststr << fmname << "(";
      moffset += ind.get() + objcaststr.length() + fmname.length() + 15;
   }
   // reference
   else if (fm.Type()->Reftype()) {
      s << ind() << "return (void*)&" << objcaststr << fmname << "(";
      moffset += ind.get() + objcaststr.length() + fmname.length() + 16;
   }
   // struct/class
   else if (retT == 'u') {
      size_t startRetType=0;
      if (!strncmp(retname.c_str(), "const ", 6)) 
         startRetType=6;
      s << ind() << "return new " << retname.substr(startRetType) << "(" 
        << objcaststr << fmname << "(";
      moffset +=
          ind.get() + retname.length() + objcaststr.length() +
          fmname.length() + 13;
   }
   // void
   else if (retT == 'y') {
      s << ind() << objcaststr << fmname << "(";
      moffset += ind.get() + objcaststr.length() + fmname.length() + 1;
   }
   // fundamental
   else {
      s << ind() << "static " << rflx_tools::
          stub_type_name(retname) << " ret" << argNum << ";" << std::endl;
      s << ind() << "ret" << argNum << " = " << objcaststr << fmname <<
          "(";
      moffset += ind.get() + objcaststr.length() + fmname.length() + 7;
   }
   return moffset;
}


// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_stubfuncdecl_params(std::ostringstream & s,
                                          G__MethodInfo & fm, int argNum)
{
   G__MethodArgInfo ma(fm);
   int maNum = 0;
   if (argNum < 0)
      argNum = 9999;
   while (ma.Next() && (maNum < argNum)) {
      // from second line on
      if (maNum)
         s << "," << std::endl << ind();
      std::string pStr;
      std::string cvStr;
      std::string arrStr;
      const char* posArr=0;
      if (ma.Name() && (posArr = strchr(ma.Name(),'['))) {
         // first array dim is "*"
         arrStr = "*";
         posArr = strchr(posArr+1, '[');
         if (posArr) {
            // multi-dim array, format as (*)[2nd][3rd][..]
            arrStr="(*)";
            arrStr+=posArr;
         }
         posArr="a"; // for pStr
      }
      // arg type IS NOT a pointer nor an array
      if (!(ma.Property() & G__BIT_ISPOINTER) && !posArr) 
         pStr = "*";
      if ((ma.Property() & G__BIT_ISREFERENCE))
         pStr = "*";
      if (ma.Property() & G__BIT_ISCONSTANT)
         cvStr += "const ";
      s << pStr << "(" << cvStr;
      if (ma.Type()->Name() && strstr(ma.Type()->Name(),"(*)"))
         // func ptr
         s << ma.Type()->Name() << arrStr;
      else if ((! ma.Type()->Fullname() &&
                strstr(ma.Type()->TrueName(),"void*") && 
                strcmp(ma.Type()->Name(),"void*"))
         || !strcmp(ma.Type()->TrueName(),"G__p2memfunc"))
      //else if (ma.Type()->Type()=='a')
         // func ptr with typedef
         s << rflx_tools::stub_type_name(ma.Type()->Name()) << arrStr << pStr;
      else
         s << rflx_tools::stub_type_name(ma.Type()->TrueName()) << arrStr << pStr;
      s << ")arg[" << maNum << "]";
      ++maNum;
   }
}


// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_stubfuncdecl_trailer(std::ostringstream & s,
                                           G__MethodInfo & fm, int argNum)
{
   if (argNum < 0)
      argNum = 0;
   int index = fm.Index();
   G__ifunc_table_internal *var = G__get_ifunc_internal((G__ifunc_table *) fm.Handle());
   G__SIGNEDCHAR_T retT = var->type[index];
   // reference 
   if (fm.Type()->Reftype()) {
      s << ");" << std::endl;
   }
   // struct/class
   else if (retT == 'u') {
      s << "));" << std::endl;
   }
   // void
   else if (retT == 'y') {
      s << ");" << std::endl << ind() << "return 0;" << std::endl;
   }
   // pointer
   else if (isupper(retT)) {
      s << ");" << std::endl;
   }
   // fundamental
   else {
      s << ");" << std::endl
          << ind() << "return &ret" << argNum << ";" << std::endl;
   }
}


// ______________________________________________________________________________
/* Generate the declarations of a dictionary class (except the constructor), i.e.
* the implementations of the stub functions.
*/
void rflx_gensrc::gen_classdictdecls(std::ostringstream & s,
                                     G__ClassInfo & ci)
{
   // M: auto generated structors
   // M: default parameter handling
   ind.clear();
   std::string clname = ci.Name();
   //std::string fclname = rflx_tools::decorate_stl_type(ci.Fullname());
   std::string fclname = ci.Fullname();
   if (!strncmp(fclname.c_str(),"pair",4))
      fclname.insert(0,"std::");

   std::string cldname =
       "__" + rflx_tools::escape_class_name(fclname) + "_dict";
   s << ind() << "//" << std::endl;
   s << ind() << "// ---------- Stub functions for class " << clname <<
       " ----------" << std::endl;
   s << ind() << "//" << std::endl;
   int mNum = -1;
   int cNum = -1;
   G__MethodInfo fm(ci);
   while (fm.Next()) {
      if (!(fm.Property() & G__BIT_ISPUBLIC))
         continue;
      std::string fmname = fm.Name();
      if (fmname.length() && (fmname[0] != '~')) {      // Q: a funny function with no name, what is it?
         int moffset = 0;
         int nDefaultArgs = fm.NDefaultArg();
         int nArgs = fm.NArg();
         bool isConstructor = false;
         if (fmname == clname)
            isConstructor = true;

         if (isConstructor) {
            if (nArgs > 0) {
               if (ci.Property() & G__BIT_ISABSTRACT) {
                  s << ind() << "void* " << cldname << "::" << "constructor_" <<
                   ++cNum <<
                   "(void*, const std::vector<void*>&, void*) {" <<
                   std::endl;
               } else {
                  s << ind() << "void* " << cldname << "::" << "constructor_" <<
                   ++cNum <<
                   "(void* mem, const std::vector<void*>& arg, void*) {" <<
                   std::endl;
               }
            } else {
               if (ci.Property() & G__BIT_ISABSTRACT) {
                  s << ind() << "void* " << cldname << "::" << "constructor_" <<
                   ++cNum <<
                   "(void*, const std::vector<void*>& , void*) {" <<
                   std::endl;
               } else {
                  s << ind() << "void* " << cldname << "::" << "constructor_" <<
                   ++cNum <<
                   "(void* mem, const std::vector<void*>& , void*) {" <<
                   std::endl;
               }
            }
            ++ind;
            if (ci.Property() & G__BIT_ISABSTRACT) {
               s << "  return 0; // pure virtual" << std::endl
                 << "}" << std::endl << std::endl;
               continue;
            }
         } else {
            s << ind() << "void* " << cldname << "::method_" <<
               ++mNum << "(void*";
            if (!(fm.Property() & G__BIT_ISSTATIC)) s << " o";
            if (nArgs > 0) 
               s << ", const std::vector<void*>& arg, void*) {" <<
                   std::endl;
            else
               s << ", const std::vector<void*>& , void*) {" <<
                   std::endl;
            ++ind;
         }

         if (nDefaultArgs) {
            bool first = true;
            for (int j = nArgs - nDefaultArgs; j <= nArgs; ++j) {
               if (first)
                  s << ind();
               else
                  s << ind() << "else ";
               s << "if (arg.size() == " << j << ") {" << std::endl;
               ++ind;

               // the part before the parameters
               if (isConstructor) {
                  s << ind() << "return ::new(mem) ::" << fclname << "(";
                  moffset += 21 + fclname.length() + ind.get();
               } else {
                  std::string constStr;
                  if  (fm.Type()->Isconst()&G__CONSTFUNC) constStr = "const ";
                  std::string objcaststr = "((" + constStr + "::" + fclname + "*)o)->";
                  if (fm.Property() & G__BIT_ISSTATIC)
                     objcaststr = fclname + "::";
                  moffset = gen_stubfuncdecl_header(s, fm, objcaststr, j);
               }
               // handle function parameters
               int oldIndex = ind.get();
               ind.set(moffset);
               gen_stubfuncdecl_params(s, fm, j);
               ind.set(oldIndex);
               // the part after the parameters
               if (isConstructor) {
                  s << ");" << std::endl;
               } else {
                  gen_stubfuncdecl_trailer(s, fm, j);
               }

               --ind;
               s << ind() << "}" << std::endl;
               first = false;
            }

         } else {

            // the part before the parameters
            if (isConstructor) {
               s << ind() << "return ::new(mem) ::" << fclname << "(";
               moffset += 21 + fclname.length() + ind.get();
            } else {
               std::string constStr;
               if  (fm.Type()->Isconst()&G__CONSTFUNC) constStr = "const ";
               std::string objcaststr = "((" + constStr + "::" + fclname + "*)o)->";
               if (fm.Property() & G__BIT_ISSTATIC)
                  objcaststr = fclname + "::";
               moffset = gen_stubfuncdecl_header(s, fm, objcaststr);
            }
            // handle function parameters
            int oldIndex = ind.get();
            ind.set(moffset);
            gen_stubfuncdecl_params(s, fm);
            ind.set(oldIndex);
            // the part after the parameters
            if (isConstructor) {
               s << ");" << std::endl;
            } else {
               gen_stubfuncdecl_trailer(s, fm);
            }

         }

         if (nDefaultArgs)
            s << ind() << "return 0;" << std::endl;
         --ind;
         s << "}" << std::endl << std::endl;
      }
   }

   // generate method __getBasesTable returning base classes
   s << ind() << "void* " << cldname 
      << "::method_getBaseTable( void*, const std::vector<void*>&, void*)" << std::endl
      << ind() << "{" << std::endl;
   ++ind;
   s << ind() << "static std::vector<std::pair<ROOT::Reflex::Base, int> > s_bases;" << std::endl;

   int iBase=0;
   std::list<int> baseTags;
   std::list<std::pair<G__ClassInfo,std::pair<int /*access*/, int /*inhlevel*/> > > baseClassesToSearch;
   baseClassesToSearch.push_back(std::make_pair(ci, std::make_pair(G__BIT_ISPUBLIC,0)));
   while (baseClassesToSearch.size()) {
      G__BaseClassInfo ciBase(baseClassesToSearch.front().first);
      int accessBase=baseClassesToSearch.front().second.first;
      int inhlevel = baseClassesToSearch.front().second.second;
      while (ciBase.Next()) {
         if (  (ciBase.Property() & G__BIT_ISVIRTUALBASE) &&
              !(ciBase.Property() & G__BIT_ISDIRECTINHERIT)) {
            // CINT duplicates the remote virtual base class in the list scanned
            // by G__BaseClassInfo, we need to skip them.
            continue;
         }
         if (std::find(baseTags.begin(), baseTags.end(), ciBase.Tagnum())!=baseTags.end())
            continue; // already dealt with

         baseTags.push_back(ciBase.Tagnum());
         G__ClassInfo ciBaseClass(ciBase);
         int myAccess=ciBase.Property() & 
            (G__BIT_ISVIRTUALBASE | G__BIT_ISPRIVATE | G__BIT_ISPROTECTED | G__BIT_ISPUBLIC);
         if ((myAccess & G__BIT_ISPROTECTED) && (accessBase & G__BIT_ISPRIVATE))
            myAccess=(myAccess & ~G__BIT_ISPROTECTED) | G__BIT_ISPRIVATE;
         else if (myAccess & G__BIT_ISPUBLIC) {
            if (accessBase & G__BIT_ISPRIVATE) 
               myAccess=(myAccess & ~G__BIT_ISPUBLIC) | G__BIT_ISPRIVATE;
            else if (accessBase & G__BIT_ISPROTECTED) 
               myAccess=(myAccess & ~G__BIT_ISPUBLIC) | G__BIT_ISPROTECTED;
         }

         baseClassesToSearch.push_back(std::make_pair(ciBaseClass, std::make_pair(myAccess,inhlevel+1)));

         std::string access;
         if (myAccess & G__BIT_ISVIRTUALBASE)
            access = "VIRTUAL|";
         if (myAccess & G__BIT_ISPRIVATE)
            access += "PRIVATE";
         else if (myAccess & G__BIT_ISPROTECTED)
            access += "PROTECTED";
         else if (myAccess & G__BIT_ISPUBLIC)
            access += "PUBLIC";

         std::string basetype_name=ciBase.Fullname();

         if (!iBase) {
            s << ind() << "if ( !s_bases.size() ) {" << std::endl;
            ++ind;
            iBase++;
         }

         s << ind() << "s_bases.push_back(std::make_pair(ROOT::Reflex::Base( ROOT::Reflex::GetType< " 
            << basetype_name << " >(), ROOT::Reflex::BaseOffset< " << fclname << "," << basetype_name 
            << " >::Get()," << access << "), " << inhlevel << "));" << std::endl;
      }
      baseClassesToSearch.erase(baseClassesToSearch.begin());
  }

  --ind;
  if (iBase) s << ind() << "}" << std::endl;

  s << ind() << "return &s_bases;" << std::endl;
  --ind;
  s << ind() << "}" << std::endl << std::endl;
}


// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_freefundicts()
{
   std::ostringstream stub_decl, type_defn, stub_defn;

   ind.clear();
   std::string cldname =
       "__reflex__free__functions__dict__" +
       rflx_tools::escape_class_name(m_sourcefile);
   int mNum = 0;
   G__MethodInfo mi;
   while (mi.Next()) {

      std::string fmname = mi.Name();
      G__ifunc_table_internal *ifunc = G__get_ifunc_internal((G__ifunc_table *) mi.Handle());

      if ((fmname.length()) 
         && ifunc->globalcomp[mi.Index()] == G__CPPLINK) {
         // stub function declarations
         ++ind;
         stub_decl << ind() << "static void * freefunction_" << mNum <<
             "(void*, const std::vector<void*>&, void*);" << std::endl;
         --ind;

         std::string fun_ns = "";
         G__ClassInfo cl(mi.ifunc()->tagnum);
         if (cl.Property() & G__BIT_ISNAMESPACE)
            fun_ns += std::string(cl.Name()) + "::";

         // function definitions
         std::string fmodifiers = "PUBLIC";
         std::string retParTypes = gen_type(*mi.Type());
         std::string parNames = "";
         gen_parTypesNames(retParTypes, parNames, mi);
         ++ind;
         type_defn << ind() << "Type ft" << mNum <<
             " = FunctionTypeBuilder(" << retParTypes << "); " <<
             "FunctionBuilder(ft" << mNum << ", \"" << fun_ns << fmname <<
             "\", freefunction_" << mNum << ", 0, \"" << parNames << "\", "
             << fmodifiers << ");" << std::endl;
         --ind;


         // stub function defintions
         int nDefaultArgs = mi.NDefaultArg();
         int nArgs = mi.NArg();
         if (nArgs > 0) {
            stub_defn << ind() << "void* " << cldname << "::" <<
             "freefunction_" << mNum <<
             "(void*, const std::vector<void*>& arg, void*) {" << std::
             endl;
         } else {
            stub_defn << ind() << "void* " << cldname << "::" <<
             "freefunction_" << mNum <<
             "(void*, const std::vector<void*>& , void*) {" << std::
             endl;
         }
         ++ind;

         if (nDefaultArgs) {
            bool first = true;
            for (int j = nArgs - nDefaultArgs; j <= nArgs; ++j) {
               if (first)
                  stub_defn << ind();
               else
                  stub_defn << ind() << "else ";
               stub_defn << "if (arg.size() == " << j << ") {" << std::
                   endl;
               ++ind;

               // the part before the parameters
               int moffset =
                   gen_stubfuncdecl_header(stub_defn, mi, fun_ns, j);
               // handle parameters
               int oldIndex = ind.get();
               ind.set(moffset);
               gen_stubfuncdecl_params(stub_defn, mi, j);
               ind.set(oldIndex);
               // the part after the parameters
               gen_stubfuncdecl_trailer(stub_defn, mi, j);

               --ind;
               stub_defn << ind() << "}" << std::endl;
               first = false;
            }

         } else {

            // the part before the parameters
            int moffset = gen_stubfuncdecl_header(stub_defn, mi, fun_ns);
            // handle parameters
            int oldIndex = ind.get();
            ind.set(moffset);
            gen_stubfuncdecl_params(stub_defn, mi);
            ind.set(oldIndex);
            // the part after the parameters
            gen_stubfuncdecl_trailer(stub_defn, mi);

         }
         --ind;
         if (nDefaultArgs)
            stub_defn << ind() << "return 0;" << std::endl;
         stub_defn << "}" << std::endl << std::endl;


         ++mNum;

      }
   }

   m_ff << ind() << "//" << std::endl;
   m_ff << ind() <<
       "// ---------- Dictionary for free functions ----------" << std::
       endl;
   m_ff << ind() << "//" << std::endl;
   m_ff << ind() << "class " << cldname << " {" << std::endl;
   m_ff << ind() << "public:" << std::endl;
   ++ind;
   m_ff << ind() << cldname << "();" << std::endl;
   m_ff << stub_decl.str();
   --ind;
   m_ff << ind() << "};" << std::endl << std::endl;
   m_ff << ind() << cldname << "::" << cldname << "() {" << std::endl;
   ++ind;
   m_ff << type_defn.str();
   --ind;
   m_ff << ind() << "}" << std::endl << std::endl;
   m_ff << ind() << "//" << std::endl;
   m_ff << ind() <<
       "// ---------- Stub functions for free functions ----------" <<
       std::endl;
   m_ff << ind() << "//" << std::endl;
   m_ff << stub_defn.str();

}



// ______________________________________________________________________________
/*
*
*/
void rflx_gensrc::gen_freevardicts()
{
   ind.clear();
   std::string cldname =
       "__reflex__free__variables__dict__" +
       rflx_tools::escape_class_name(m_sourcefile);
   m_fv << ind() << "//" << std::endl;
   m_fv << ind() <<
       "// ---------- Dictionary for free variables ----------" << std::
       endl;
   m_fv << ind() << "//" << std::endl;
   m_fv << ind() << "class " << cldname << " {" << std::endl;
   m_fv << ind() << "public:" << std::endl;
   ++ind;
   m_fv << ind() << cldname << "();" << std::endl;
   --ind;
   m_fv << ind() << "};" << std::endl << std::endl;
   m_fv << ind() << cldname << "::" << cldname << "() {" << std::endl;
   ++ind;
   G__DataMemberInfo var;
   std::list<G__DataMemberInfo> cppMacros;

   while (var.Next()) {
      if (var.MemberOf() && var.MemberOf()->IsValid())
         continue;

      G__var_array* varArr = (G__var_array*)var.Handle();
      if (varArr->globalcomp[var.Index()] != G__CPPLINK) continue;

      if (var.Type() && strchr("pT", var.Type()->Type())) {
         // CPP macro
         cppMacros.push_back(var);
         continue;
      }

      long dmProp = var.Property();
      if (dmProp & (G__BIT_ISENUM | G__BIT_ISSTATIC | G__BIT_ISCONSTANT) && var.Type()) {
         // enum const
         gen_type(*var.Type());
         continue;
      }

      std::string modifiers;
      if (dmProp & G__BIT_ISSTATIC)
         modifiers += "STATIC";

      if (!modifiers.empty()) modifiers.insert(0, ", ");
      m_fv << ind() << "VariableBuilder( \"" << var.Name() << "\", "
           << gen_type(*var.Type()) << ", (size_t)&" << var.Name() << modifiers << " );" << std::endl;
   }

   // setup CPP macros
   if (cppMacros.size()) {
      m_fv << ind() << "Scope cppMacroEnum = Scope::ByName(\"$CPP_MACROS\");" << std::endl
         << ind() << "Type typeCPPMacro = Type::ByName(\"$CPPMACRO\");" << std::endl
         << ind() << "if (!cppMacroEnum || !cppMacroEnum.IsEnum()) {" << std::endl
         << ind() << "  typedef void CPP_MACRO_DUMMY_TYPE;" << std::endl 
         << ind() << "  EnumBuilder(\"$CPP_MACROS\", typeid(CPP_MACRO_DUMMY_TYPE));" << std::endl
         << ind() << "  cppMacroEnum = Type::ByName(\"$CPP_MACROS\");" << std::endl
         << ind() << "  typeCPPMacro = TypeBuilder(\"$CPPMACRO\");" << std::endl
         << ind() << "}" << std::endl;
      for (std::list<G__DataMemberInfo>::iterator iCPP=cppMacros.begin();
         iCPP!=cppMacros.end(); ++iCPP) {
            m_fv << ind() << "cppMacroEnum.AddDataMember(\"" << iCPP->Name()
               << "\", typeCPPMacro, (size_t)\"";
            G__var_array* va = (G__var_array*)iCPP->Handle();
            long index=iCPP->Index();
            if (va->type[index] == 'p') m_fv << *(int*)va->p[index];
            else m_fv << *(const char**)va->p[index];
            m_fv << "\", ARTIFICIAL);" << std::endl;
      }
   }

   --ind;
   m_fv << ind() << "}" << std::endl << std::endl;
}

// ______________________________________________________________________________
/* Generate the code to invoke the dictionary classes with a static invocation
*/
void rflx_gensrc::gen_dictinstances()
{
   ind.clear();
   m_di << ind() << "//" << std::endl;
   m_di << ind() << "// ---------- Dictionary instantiations ----------" <<
       std::endl;
   m_di << ind() << "//" << std::endl;
   m_di << ind() << "namespace {" << std::endl;
   ++ind;
   m_di << ind() << "struct _Dictionaries { " << std::endl;
   ++ind;
   m_di << ind() << "_Dictionaries() {" << std::endl;
   ++ind;
   m_di << ind() << "ROOT::Cintex::Cintex::Enable();" << std::endl;
   m_di << "#if defined (CINTEX_DEBUG)" << std::endl;
   m_di << ind() << "ROOT::Cintex::Cintex::SetDebug(1);" << std::endl;
   m_di << "#endif" << std::endl;
   m_di << ind() << "__reflex__free__functions__dict__" 
        << rflx_tools::escape_class_name(m_sourcefile) << "();" << std::endl;
   m_di << ind() << "__reflex__free__variables__dict__" 
        << rflx_tools::escape_class_name(m_sourcefile) << "();" << std::endl;
   for (std::vector < std::string >::const_iterator it =
        m_classNames.begin(); it != m_classNames.end(); ++it) {
      m_di << ind() << "__" << rflx_tools::
          escape_class_name(*it) << "_dict();" << std::endl;
   }
   // enums at the end!
   m_di << ind() << "__reflex__enums__dict__" 
        << rflx_tools::escape_class_name(m_sourcefile) << "();" << std::endl;
   --ind;
   m_di << ind() << "}" << std::endl;
   --ind;
   m_di << ind() << "};" << std::endl;

   // generate an external C func called "G__cpp_setup"dictfile, to be compatible with cint
   std::string staticName=m_dictfile;
   size_t posExt=staticName.find('.');
   if (posExt!=std::string::npos)
      staticName.erase(posExt);
   size_t posDirEnd=staticName.rfind('/');
   size_t posDirEndBackSlash=staticName.rfind('\\');
   if (posDirEndBackSlash!=std::string::npos && posDirEnd<posDirEndBackSlash)
      posDirEnd=posDirEndBackSlash;
   if (posDirEnd!=std::string::npos)
      staticName.erase(0,posDirEnd+1);

   m_di << ind() << "static _Dictionaries G__cpp_setup" << staticName << "_dict;"  << std::endl;

   --ind;
   m_di << ind() << "}" << std::endl;

   m_di << ind() << "extern \"C\" void G__cpp_setup" << staticName << "(void) {}" << std::endl;
}
