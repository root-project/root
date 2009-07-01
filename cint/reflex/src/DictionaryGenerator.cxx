// @(#)root/reflex:$Id$
// Author: Antti Hahto   06/20/06

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif


// DictGen, a replacement for genreflex.py
//================================================


#include "Reflex/DictionaryGenerator.h"
#include "Reflex/Member.h"
#include "Class.h"

// For adding the XML parser
//#include "TDOMParser.h"
//#include "TXMLNode.h"

// USAGE:
//
// 1. Create a new generator                        :  Dictgen generator
// 2. Set recursive parsing, optional (default:on)  :  generator.Use_recursive(true/false)
// 3. Set selection file,    optional               :  generator.Use_selection("filename")
// 4. Run, example
//      Scope::GlobalScope().GenerateDict(generator);
// 5. Dump results into file/stdout(if filename left empty)  : generator.Dump("filename")


using namespace std;

//-------------------------------------------------------------------------------
std::ostream&
Reflex::operator <<(std::ostream& s,
                    const Reflex::DictionaryGenerator& obj) {
//-------------------------------------------------------------------------------

   time_t rawtime;
   time(&rawtime);

   s << "//Generated at " << ctime(&rawtime) << "//Do not modify." << endl << endl;
   s << "#include \"Reflex/Builder/ReflexBuilder.h\"" << endl;
   s << "#include <typeinfo>" << endl;
   s << "using namespace Reflex;" << endl << endl;
   s << obj.fStr_header.str();

   s << "namespace {" << endl;
   s << obj.fStr_namespaces.str();
   s << "}" << endl << endl;

   s << "// ---------------------- Shadow classes -----------------------------------" << endl;
   s << "namespace __shadow__ {" << endl;
   s << obj.fStr_shadow.str() << endl;
   s << "}" << endl << endl;

   s << "// ---------------------- Stub functions -----------------------------------" << endl;
   s << "namespace {" << endl;
   s << obj.fStr_classes.str();
   s << "} // unnamed namespace" << endl << endl;

   s << "// -------------------- Class dictionaries ---------------------------------" << endl;
   s << obj.fStr_frees.str();

   s << "// --------------------- Dictionary instances ------------------------------" << endl;
   s << "namespace {" << endl;
   s << "  struct Dictionaries {" << endl;
   s << "    Dictionaries() {" << endl;
   s << obj.fStr_instances.str();
   s << "    }" << endl;
   s << "    ~Dictionaries() {" << endl;
   s << obj.fStr_instances2.str();
   s << "    }" << endl;
   s << "  };" << endl;
   s << "  static Dictionaries instance;" << endl;
   s << "}" << endl << endl;
   s << "// End of Dictionary" << endl;

   return s;

} // <<


//-------------------------------------------------------------------------------
std::string
Reflex::DictionaryGenerator::Replace_colon(std::string scoped_name) {
//-------------------------------------------------------------------------------
// Replaces :: with __ , in a string
   std::string lname = scoped_name;

   for (unsigned i = 0; i < lname.size(); ++i) {
      switch (lname[i]) {
      case '<':
      case '>':
      case ' ':
      case ',':
      case ':':
      case '.':
      case '*':
      case '(':
      case ')':
      case '&':
         lname[i] = '_';
         break;
      default:
         break;
      }
   }
   return lname;
} // Replace_colon


//-------------------------------------------------------------------------------
bool
Reflex::DictionaryGenerator::Use_selection(const std::string& filename) {
//-------------------------------------------------------------------------------
// User defined selection file for classes to include
   std::ifstream infile;

   if (filename != "") {
      infile.open(filename.c_str());

      if (!infile.is_open()) {
         std::cout << "Error: Selection file not found!\n";
         infile.clear();
         return false;
      }

      std::cout << "\nUsing selection file:\n";
      std::string line = "";

      while (getline(infile, line)) {
         if (line.find("class name") != std::string::npos) {
            size_t start = line.find("\"");
            size_t end = line.rfind("\"/>");
            // cut the class name out of string
            line = line.substr(start + 1, end - start - 1);
            fSelections.push_back(line);
            std::cout << "searching for class " << line << "\n";
         }

         // <class pattern="Reflex::*"/>
         if (line.find("class pattern") != std::string::npos) {
            size_t start = line.find("=");
            size_t end = line.rfind("*");
            line = line.substr(start + 2, end - start - 2);
            fPattern_selections.push_back(line);
            std::cout << "searching for class pattern " << line << "\n";
         }

      }


      /*   THIS IS HOWTO USE THE EXISTING XML-PARSER IN ROOT, TESTING

         TDOMParser parseri;
         parseri.SetValidate(kFALSE);

         if( parseri.ParseFile("/home/ahahto/root2/test_selection.xml") !=0)
         {
         std::cerr<< "Error: no valid lcgdict XML file.\n";
         // return EXIT_FAILURE;
         }

         TXMLDocument *document = parseri.GetXMLDocument();
         TXMLNode *rootnode = document->GetRootNode();

         std::string rootnodename =  rootnode->GetNodeName();
         std::cout<<"Rootnode: ["<< rootnodename <<"]\n";

         if( rootnodename != "lcgdict")
         {
         std::cerr<< "Error: no valid lcgdict XML file.\n";
         // return EXIT_FAILURE;
         }

         //  TXMLNode *childnode = node->GetChildren();

         for ( TXMLNode *childnode = rootnode->GetChildren(); childnode !=0;  childnode = childnode ->GetNextNode())
         {
         std::cout<< "\n* " << childnode->GetNodeName();

         if(childnode !=0)
         {

         TList* lista =  childnode->GetAttributes();
         if(lista!=0)
         {
         //TObject* objekti = lista -> First();
         for(TObject* objekti = lista -> First(); objekti !=0; objekti = lista -> After(objekti) )
         {
         if(objekti!=0)
         {
         std::cout<< " [" << objekti ->GetName() <<"]";
         }
         }
         }
         }

         }
       */


   }

   return true;
} // Use_selection


//-------------------------------------------------------------------------------
bool
Reflex::DictionaryGenerator::Use_recursive(bool recursive) {
//-------------------------------------------------------------------------------
// Set the usage of recursive selection

   if (recursive == false) {
      fSelect_recursive = false;
   }

   if (recursive == true) {
      fSelect_recursive = true;
   }

   return true;
}


//-------------------------------------------------------------------------------
bool
Reflex::DictionaryGenerator::Use_recursive() {
//-------------------------------------------------------------------------------
// Returns, if selected recursive selection

   return fSelect_recursive;

}


//-------------------------------------------------------------------------------
bool
Reflex::DictionaryGenerator::IsNewType2(const Type& searchtype) {
//-------------------------------------------------------------------------------
// An optimized IsNewType, TESTING

/*   std::list < mydata >::iterator i =
      std::find_if(types2.begin(), types2.end(),
                   MatchItemNum(searchtype));

   if (i == types2.end()) {
      // not found
      return true;
   }

   else {
      //found
      return false;
   }
 */

// dummy return

   return searchtype;

}


//-------------------------------------------------------------------------------
bool
Reflex::DictionaryGenerator::IsNewType(const Type& searchtype) {
//-------------------------------------------------------------------------------
// Is the type already used somewhere
// This is used by GetTypeNumber

   /*  std::vector<Reflex::Type>::iterator start = fTypes.begin();
       std::vector<Reflex::Type>::iterator end = fTypes.end();
       std::vector<Reflex::Type>::iterator iter;


       // pred is a comparing function used, IsEqual
       iter = find_if(start,end, searchtype, pred );


       if (iter == end)
       {
       // not found
       return true;

       }

       else
       {

       //found
       return false;
       }
    */

   for (unsigned k = 0; k < fTypes.size(); ++k) {
      if (fTypes.at(k) == searchtype) {
         return false;
      }
   }

   return true;
} // IsNewType


//-------------------------------------------------------------------------------
std::string
Reflex::DictionaryGenerator::GetParams(const Type& membertype) {
//-------------------------------------------------------------------------------
// Get params for the functions
// Recursive, if it's a reference or a pointer
   if (membertype.IsReference()) {
      GetParams(membertype.ToType());
   }

   if (membertype.IsPointer()) {
      GetParams(membertype.ToType());
   }

   bool newtype = IsNewType(membertype);

   if (newtype) {
      // we add a new datamember not occurred before
      fTypes.push_back(membertype);

      //      types2.push_back(membertype);  //TESTING

      if (!membertype.IsFunction()) {
         // adding into NS this way
         GetTypeNumber(membertype);
      }
   }

   return membertype.Name();
} // GetParams


//-------------------------------------------------------------------------------
std::string
Reflex::DictionaryGenerator::GetTypeNumber(const Type& membertype) {
//-------------------------------------------------------------------------------
// If a type is registered (eg.used) before,
// it has an unique ID-number returned as a string
// Also, add this type into NS field

   bool newtype = IsNewType(membertype);

   if (newtype) {
      // we add a new datamember not occurred before
      fTypes.push_back(membertype);
      //      types2.push_back(membertype);
   }

   std::stringstream numberstr;

   // get number
   for (unsigned k = 0; k < fTypes.size(); ++k) {
      if (fTypes.at(k) == membertype) {
         numberstr << k;
      }
   }

   // when it's a new type, add also into NS field
   // (No function members added)
   if (newtype && !(membertype.IsFunction())) {
      AddIntoNS(numberstr.str(), membertype);
   }

   return numberstr.str();
} // GetTypeNumber


// GetTypeNumber


//-------------------------------------------------------------------------------
void
Reflex::DictionaryGenerator::AddIntoNS(const std::string& typenumber,
                                       const Type& membertype) {
//-------------------------------------------------------------------------------
// Namespaces field of the generated file

   if (fStr_namespaces.str() == "0") {
      //first item of the stream

      fStr_namespaces << "\nnamespace {  \n";
      fStr_namespaces << "Type type_void = TypeBuilder(\"void\");\n";
   }
   // Name(SCOPED)  adds also the namespace into name

   //     forward declarations
   if (membertype.TypeType() == STRUCT || membertype.TypeType() == CLASS
       || membertype.TypeType() == TYPEDEF) {
      fStr_shadow2 << (membertype).Name(SCOPED) << ";\n";
   }


   unsigned mod = 0;

   // if type is also a const, add that extra line
   // No references are supported by the original genreflex.py for datamembers!
   if (membertype.IsReference()) {
      if (membertype.IsConst()) {
         mod |= CONST;
      }

      if (membertype.IsVolatile()) {
         mod |= VOLATILE;
      }

      fStr_namespaces << "Type type_" + typenumber + " = ReferenceBuilder(type_" +
      GetTypeNumber(Type(membertype, mod)) + ");\n";
   } else if (membertype.IsConst()) {
      if (membertype.IsVolatile()) {
         mod |= VOLATILE;
      }

      fStr_namespaces << "Type type_" + typenumber +
      " = ConstBuilder(type_" + GetTypeNumber(Type(membertype, mod)) + ");\n";

   } else if (membertype.IsVolatile()) {
      fStr_namespaces << "Type type_" + typenumber + " = VolatileBuilder(type_" +
      GetTypeNumber(Type(membertype, mod)) + ");\n";

   } else if (membertype.TypeType() == CLASS) {
      fStr_namespaces << "Type type_" + typenumber + " = TypeBuilder(\"" + membertype.Name(SCOPED) + "\"); //class\n";
      //membertype
      fStr_instances2 << ("    type_" + typenumber + ".Unload(); //class " + (membertype).Name(SCOPED) + "\n");

   } else if (membertype.IsPointer()) {
      fStr_namespaces << "Type type_" + typenumber +
      " = PointerBuilder(type_" + GetTypeNumber(membertype.ToType()) + ");\n";
   }
   // void type
   else if (membertype.Name(SCOPED) == "") {
   } else {
      fStr_namespaces << "Type type_" + typenumber + " = TypeBuilder(\"" + membertype.Name(SCOPED) + "\");\n";

   }


} //AddIntoNS


//-------------------------------------------------------------------------------
void
Reflex::DictionaryGenerator::Print(const std::string& filename) {
//-------------------------------------------------------------------------------
// Outputs the results into standard stream / file

   if (filename.length()) {
      std::ofstream outfile(filename.c_str(), std::ios_base::out);

      if (!outfile.is_open()) {
         std::cout << "Error: Unable to write file!\n";
         outfile.clear();
      } else {
         outfile << *this;
         outfile.close();
      }


   } else {
      std::cout << "\n\n";
      std::cout << *this;

   }

} // Print
