#include "TClassEdit.h"
#include "TClass.h"
#include "TError.h"
#include "TInterpreter.h"
#include <string>

bool checkResult(const std::string &what, const std::string &output, const std::string &expected)
{
   if (output != expected) {
      Error("execString","When calling %s expected %s and got %s",
            what.c_str(), expected.c_str(), output.c_str());
      return false;
   }
   return true;
}

int execString()
{
   Int_t badresult = 0;
   std::string output;
   TClass *cl = 0;

   gInterpreter->GetInterpreterTypeName("list<string>",output,true);
   badresult += !checkResult("GetInterpreterTypeName",output,"list<string>");

   output = TClassEdit::ShortType("list<basic_string<char,char_traits<char>,allocator<char> >",0);
   badresult += !checkResult("ShortType",output,"list<string>");

   output = TClassEdit::ShortType("basic_string<char,char_traits<char> >",false);
   badresult += !checkResult("ShortType",output,"string");

   const char *longname = " map<basic_string<char,char_traits<char> >,map<basic_string<char,char_traits<char>,std::allocator<char> >,  basic_string<char,std::char_traits<char>,allocator<char> > >";
   output = TClassEdit::ShortType(longname,false);
   badresult += !checkResult("ShortType",output,"map<string,map<string,string> >");

   cl = TClass::GetClass("basic_string<char,char_traits<char> >");
   if (cl) {
     output = cl->GetName();
     badresult += !checkResult("GetClass",output,"string");
   } else {
     Error("execString","When calling GetClass with %s, no object was returned","basic_string<char,char_traits<char> >");
     ++badresult;
   }

   cl = TClass::GetClass("string");
   if (cl) {
     output = cl->GetName();
     badresult += !checkResult("GetClass",output,"string");
   } else {
     Error("execString","When calling GetClass with %s, no object was returned","string");
     ++badresult;
   }

   gInterpreter->Declare("namespace std { inline namespace __cxx11 {} } ");

   longname = "std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > >";
   output = TClassEdit::ShortType(longname,TClassEdit::kDropAllDefault);
   badresult += !checkResult("ShortType",output,"std::vector<std::string>");

   cl = TClass::GetClass(longname);
   if (cl) {
     output = cl->GetName();
     badresult += !checkResult("GetClass",output,"vector<string>");
     if (!cl->IsLoaded()) {
       Error("execString","When calling GetClass with %s, the TClass is in state %d",longname, cl->GetState());
       ++badresult;
     }
   } else {
     Error("execString","When calling GetClass with %s, no object was returned",longname);
     ++badresult;
   }

   return badresult;
}
