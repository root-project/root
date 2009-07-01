// @(#)root/reflex:$Id$
// Author: Antti Hahto   06/20/06

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

// DictGen, a replacement for genreflex.py
//================================================

//USAGE:
//
// 1. Create a new generator                        :  Dictgen generator
// 2. Set recursive parsing, optional (default:on)  :  generator.Use_recursive(true/false)
// 3. Set selection file,    optional               :  generator.Use_selection("filename")
// 4. Run, example
//      Scope::GlobalScope().Gendict(generator);
// 5. Dump results into file/stdout(if filename left empty)  : generator.Dump("filename")

#ifndef Reflex_DictionaryGenerator
#define Reflex_DictionaryGenerator

#include "Reflex/Kernel.h"
#include "Reflex/Type.h"

#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>
#include <ostream>
#include <sstream>
#include <list>   // isnewtype2
#include <algorithm> //isnewtype2


#ifdef _WIN32
# pragma warning( push )
# pragma warning( disable : 4251 )
#endif

namespace Reflex {
// forward declarations
class Type;

/*
 * @class DictionaryGenerator DictonaryGenerator.h Reflex/DictionaryGenerator.h
 * @author Antti Hahto
 * @date 20/06/2006
 * @ingroup Ref
 */

class RFLX_API DictionaryGenerator {
public:
   /* default constructor */
   DictionaryGenerator();


   /* destructor */
   ~DictionaryGenerator();

   friend std::ostream& operator <<(std::ostream& s,
                                    const DictionaryGenerator& obj);


   /**
    * Use_recursive set recursive or not
    * @param recursive
    * @return
    */
   bool Use_recursive(bool recursive);


   /**
    * Use_selection set selection file
    * @param filename
    * @return
    */
   bool Use_selection(const std::string& filename);


   /**
    * Dump output the results into stream
    * @param filename
    */
   void Print(const std::string& filename = "");


   /**
    * AddHeaderFile adds an extra user .h file into dump
    * @param filename
    */
   void AddHeaderFile(const std::string& filename);


   /**
    * GetTypeNumber allt types have a unique id-number
    * @param mebertype
    * @return
    */
   std::string GetTypeNumber(const Type& membertype);


   /**
    * Use_recursive get if recursion set
    * @return true if recursion is set
    */
   bool Use_recursive();


   /**
    * AddIntoInstances add Instances -field into output
    * @param item
    */
   void AddIntoInstances(const std::string& item);


   /**
    * AddIntoNS add NS field into output
    * @param typenumber
    * @param membertype
    */
   void AddIntoNS(const std::string& typenumber,
                  const Type& membertype);


   /**
    * AddIntoShadow add Shadow field into output
    * param item
    */
   void AddIntoShadow(const std::string& item);


   /**
    * AddIntoFree add Free field into output
    * @param item
    */
   void AddIntoFree(const std::string& item);


   /**
    * AddIntoClasses add Classes field into output
    * @param item
    */
   void AddIntoClasses(const std::string& item);


   /**
    * fMethodCountermethod_Xn, after Stub Functions for the class
    */
   double fMethodCounter;


   /**
    * fStr_namespaces
    */
   std::ostringstream fStr_namespaces;


   /**
    * GetParams
    * @param membertype
    */
   std::string GetParams(const Type& membertype);


   /**
    * IsNewType already introduced type?
    * @param searchtype
    * @return
    */
   bool IsNewType(const Type& searchtype);


   /**
    * Replace_colon
    * @param scoped_name
    * @return
    */
   std::string Replace_colon(std::string scoped_name);


   /**
    * fSelections
    */
   std::vector<std::string> fSelections;      // for explicitly choosing classes to include into generation


   /**
    * fPattern_selections
    */
   std::vector<std::string> fPattern_selections;

   bool IsNewType2(const Type& searchtype);        // testing; work-in-progress

private:
   /**
    * GetSubScopes one scope can include multiple subscopes
    * @param allscopes
    */
   void GetSubScopes(Scope_Iterator allscopes);


   /**
    * GetMembers and subscope can include members
    */
   void GetMembers(Scope_Iterator subsco);


   /**
    * fTypes store used types
    */
   std::vector<Reflex::Type> fTypes;


   /**
    * fStr_header
    */
   std::ostringstream fStr_header;


   /**
    * fStr_shadow
    */
   std::ostringstream fStr_shadow;


   /**
    * fStr_shadow2 member predefinations for shadow
    */
   std::ostringstream fStr_shadow2;


   /**
    * fStr_classes
    */
   std::ostringstream fStr_classes;


   /**
    * fStr_classes_method for class part, method_xn
    */
   std::ostringstream fStr_classes_method;


   /**
    * fStr_frees
    */
   std::ostringstream fStr_frees;


   /**
    * fStr_instances
    */
   std::ostringstream fStr_instances;


   /**
    * fStr_instances2 instances unload() -part
    */
   std::ostringstream fStr_instances2;


   /**
    * fSelect_recursive if set true, recursive go throught all the scopes
    */
   bool fSelect_recursive;

   /*
      // FIND2 START testing
      struct mydata
      {
      Reflex::Type itemnum;
      Reflex::Type value;
      };
      struct MatchItemNum
      {
      Reflex::Type itemnum;
      MatchItemNum(Reflex::Type num) : itemnum(num)
      {
      }
      bool operator()(const mydata &data) const
      {
      //return (data.itemnum == itemnum);
      return (data.itemnum.IsEquivalentTo(itemnum));
      }
      };
      std::list<mydata> types2;
      // FIND2 END
    */

};    // class DictionaryGenerator


/** stream operator */
std::ostream& operator <<(std::ostream& s,
                          const DictionaryGenerator& obj);

} // namespace Reflex

//-------------------------------------------------------------------------------
inline void
Reflex::DictionaryGenerator::AddHeaderFile(const std::string& filename) {
//-------------------------------------------------------------------------------
// Adds an extra user .h -file into dump
   fStr_header << "#include \"" << filename << "\"\n";
}


//-------------------------------------------------------------------------------
inline void
Reflex::DictionaryGenerator::AddIntoInstances(const std::string& item) {
//-------------------------------------------------------------------------------
// The last field of the generated file
   fStr_instances << item;

}


//-------------------------------------------------------------------------------
inline void
Reflex::DictionaryGenerator::AddIntoShadow(const std::string& item) {
//-------------------------------------------------------------------------------
// Shadow field of the generated file
   fStr_shadow << item;
}


//-------------------------------------------------------------------------------
inline void
Reflex::DictionaryGenerator::AddIntoFree(const std::string& item) {
//-------------------------------------------------------------------------------
// Free field generation
   fStr_frees << item;
}


//-------------------------------------------------------------------------------
inline void
Reflex::DictionaryGenerator::AddIntoClasses(const std::string& item) {
//-------------------------------------------------------------------------------
// Classes field generation
   fStr_classes << item;
}


//-------------------------------------------------------------------------------
inline Reflex::DictionaryGenerator::DictionaryGenerator()
//-------------------------------------------------------------------------------
   : fMethodCounter(0),
   fSelect_recursive(true) {
   fTypes.clear(); // storage of used types
}


//-------------------------------------------------------------------------------
inline Reflex::DictionaryGenerator::~DictionaryGenerator() {
//-------------------------------------------------------------------------------
}


#ifdef _WIN32
# pragma warning( pop )
#endif

#endif // Reflex_DictionaryGenerator
