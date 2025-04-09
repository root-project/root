#include "TSystem.h"
#include "TClass.h"
#include "TROOT.h"
#include "TSystem.h"
#include "TObject.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TMethodArg.h"
#include "TDictAttributeMap.h"
#include <iostream>
#include <vector>
#include <string>

typedef std::vector<std::string> propertiesNames;
typedef std::map<std::string, propertiesNames > memberNamesProperties;
memberNamesProperties emptyMap;

//______________________________________________________________________________
void loadLib(const char* libname)
{
   std::cout << "Loading library " << libname << std::endl;
   gSystem->Load(libname);
}

//______________________________________________________________________________
void printMemberNames(const std::string& className, const std::string indent="", memberNamesProperties& mbProp = emptyMap)
{
   TClass* theClass = TClass::GetClass(className.c_str());
   TList* listOfDataMembers(theClass->GetListOfDataMembers());
   TIter next(listOfDataMembers);
   TDataMember* dm;
   while ( ( dm = static_cast<TDataMember*> (next()) ) ){
      std::string dmName(dm->GetName());
      std::cout << indent <<  dm->GetTrueTypeName() << " " << dmName;
      bool isPtr=dm->IsaPointer();
      bool isBasic=dm->IsBasic();
      bool isEnum=dm->IsEnum();
      bool isTransient=!dm->IsPersistent();
      std::string dmTitle (dm->GetTitle());
      if (isPtr || isEnum || isTransient || isBasic || dmTitle!="" ){
         std::cout << " [ ";
         if (isPtr) std::cout << "isPtr ";
         if (isBasic) std::cout << "isBasic ";
         if (isEnum) std::cout << "isEnum ";
         if (isTransient) std::cout << "isTransient ";
         if (dmTitle!="") std::cout << "\"" << dmTitle << "\" ";
         std::cout << "]\n";
         
      }
      else
         std::cout << std::endl;
      if (mbProp.count(dmName)!=0){
         TDictAttributeMap* attrMap = dm->GetAttributeMap();
         for (propertiesNames::iterator prop=mbProp[dmName].begin();
              prop!=mbProp[dmName].end();prop++){
            const char* propVals = attrMap->GetPropertyAsString(prop->c_str());
            std::cout << "    - " << *prop << ": " << propVals <<  std::endl;            
         }
      }
   }
}

//______________________________________________________________________________
void getMethodArgsAsStr(TMethod* method, std::string& argsAsString)
{
   TList* args = method->GetListOfMethodArgs();
   TIter next(args);
   TMethodArg* mArg;
   while ( ( mArg = static_cast<TMethodArg*> (next()) ) ){
      argsAsString+= mArg->GetTitle();
      argsAsString+= " ";
      argsAsString+= mArg->GetName();
      argsAsString+= ", ";
   }
   if (argsAsString.size()>=2)
      argsAsString.erase (argsAsString.end()-2, argsAsString.end());
}
//______________________________________________________________________________
void printMethodNames(const std::string& className,const std::string indent="")
{
   TClass* theClass = TClass::GetClass(className.c_str());
   TList* listOfMethods(theClass->GetListOfMethods());
   TIter next(listOfMethods);
   TMethod* method;

   while ( ( method = static_cast<TMethod*> (next()) ) ){
      std::cout << indent << method->GetPrototype() << std::endl;
   }
}
//______________________________________________________________________________
void printClassInfo(const std::string& className,
                    const propertiesNames& properties = propertiesNames(),
                    bool printMethods = true,
                    memberNamesProperties& mbProp = emptyMap)
{
   TClass* theClass = TClass::GetClass(className.c_str());
   if (!theClass){
      std::cerr << "ERROR: The information about " << className << " is not in TClass!\n";
      return;
   }
   std::cout << "\n--- Class " << className << std::endl;
   std::cout << "Class category: ";
   if (theClass->IsForeign()) std::cout << "foreign.\n";
   if (theClass->IsTObject()) std::cout << "TObject.\n";

   // Get the attribute map
   TDictAttributeMap* attrMap = theClass->GetAttributeMap();
   if (attrMap) {
      for (propertiesNames::const_iterator propValType=properties.begin();
           propValType!=properties.end();propValType++){
         const std::string& prop (*propValType);
         if (attrMap->HasKey(prop.c_str())){
            const char* propVals = attrMap->GetPropertyAsString(prop.c_str());
            std::cout << "  - " << prop << ": " << propVals <<  std::endl;
         } else
            std::cout << " - " << prop << " not found!\n";
      }
   }

   int classVersion ( theClass->GetClassVersion());
   std::cout << " * Version: " << classVersion;
   if (classVersion == 1) std::cout << " --> Class available to the interpreter but not Selected!";
   std::cout << std::endl;
   if (printMethods){
      std::cout << " o Methods (" << theClass->GetNmethods() << "):\n";
      printMethodNames(className,"  * ");
   }
   std::cout << " o Members (" << theClass->GetNdata() << "):\n";
   printMemberNames(className,"  * ",mbProp);
   std::cout << "--- End Class " << className << std::endl;
}
