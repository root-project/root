#include "Inspector.hh"
#include <fstream>

using namespace ROOT::Reflex;
using namespace std;

bool Inspect::Random::fgInit = false;
TRandom3 Inspect::Random::fgRndGen;

std::map<ROOT::Reflex::Scope, Inspect::InspectorGenSource::InspectorGeneratorFunc_t>
Inspect::InspectorGenSource::fgInspectors;

namespace {
   static string::size_type repl(string& s,
                                      const string& from,
                                      const string& to)
   {
      string::size_type cnt = 0;

      if(from != to && !from.empty()) {
         string::size_type pos1(0);
         string::size_type pos2(0);
         const string::size_type from_len(from.size());
         const string::size_type to_len(to.size());
         cnt = 0;

         while((pos1 = s.find(from, pos2)) != string::npos) {
            s.replace(pos1, from_len, to);
            pos2 = pos1 + to_len;
            ++cnt;
         }
      }

      return cnt;
   }
}

bool Inspect::InspectorBase::AsExpected() const {
   if (!fActual) {
      cerr << "InspectorBase::AsExpected(): Call Inspect() first!" << endl;
      return false;
   }

   bool ret = true;
   for (ResultMap_t::const_iterator iRes = fExpected.begin();
        iRes != fExpected.end(); ++iRes) {
      if (!IsEqual(iRes->first.TypeOf(), iRes->first, iRes->second,(*fActual)[iRes->first])) {
	cerr << "InspectorBase::Compare() " << iRes->first.Name(SCOPED) << endl
	     << " expected: ";
	Dump(cerr, iRes->first, iRes->second);
	cerr << "   actual: ";
	Dump(cerr, iRes->first, (*fActual)[iRes->first]);
	cerr << endl;
	ret = false;
      }
   }

   return ret;
}


void Inspect::InspectorBase::Dump(ostream& out, const Member& mem, const void* v, const char* indent /*= 0*/) const
{
   Type type = mem.TypeOf();
   std::string sIndent;
   if (indent) sIndent = indent;
   sIndent += "[";
   sIndent += mem.Name(SCOPED);
   sIndent += "] ";
   Dump(out, type, v, sIndent.c_str());
}

void Inspect::InspectorBase::Dump(ostream& out, const Type& type, const void* v, const char* indent /*= 0*/) const
{
   if (type.IsPointer()) {
      if (!v) {
         cout << (indent?indent:"") << type.Name(SCOPED) << ": 0" << endl;
         return;
      }
      v = (void*)(*(char**)v);
      std::string sIndent;
      if (indent) sIndent = indent;
      sIndent += "* ";
      Dump(out, type.ToType(), v, sIndent.c_str());
      return;
   }


   if (!type.IsFundamental()) {
      Scope scope = type;
      if (scope) {
         cout << (indent? indent : "") << type.TypeTypeAsString() << " " << type.Name(SCOPED) << ":" << endl;
         std::string sIndent;
         if (indent)
            sIndent = indent;
         sIndent += "  ";
         for (Member_Iterator iMember = scope.DataMember_Begin();
              iMember != scope.DataMember_End(); ++iMember) {
            void* vptr = ((char*)v) + iMember->Offset();
            Dump(out, *iMember, vptr, sIndent.c_str());
         }
         return;
      }
   }

   if (!v) {
      cout << (indent?indent:"") << type.Name(SCOPED) << ": 0" << endl;
      return;
   }

   if (type.IsFunction()) {
     Dump(out, type.ReturnType(), v, indent);
     return;
   }

   out << (indent?indent:"") << type.Name(SCOPED) << ": ";
   switch (Tools::FundamentalType(type)) {
   case kCHAR:         out << *(char*)v; break;
   case kSIGNED_CHAR:  out << *(signed char*)v; break;
   case kSHORT_INT:    out << *(short int*)v; break;
   case kINT:          out << *(int*)v; break;
   case kLONG_INT:     out << *(long int*)v; break;
   case kUNSIGNED_CHAR: out << *(unsigned char*)v; break;
   case kUNSIGNED_SHORT_INT: out << *(unsigned short int*)v; break;
   case kUNSIGNED_INT: out << *(unsigned int*)v; break;
   case kUNSIGNED_LONG_INT: out << *(unsigned long int*)v; break;
   case kBOOL:         out << *(bool*)v; break;
   case kFLOAT:        out << *(float*)v; break;
   case kDOUBLE:       out << *(double*)v; break;
   case kLONG_DOUBLE:  out << *(long double*)v; break;
   case kVOID:         out << *(int*)v; break;
   case kLONGLONG:     out << *(longlong*)v; break;
   case kULONGLONG:    out << *(ulonglong*)v; break;
   default:
      cerr << "InspectorBase::Dump(): type " << type.Name(SCOPED) << " not handled!" << endl;
   };
   cout << endl;
}

bool Inspect::InspectorBase::IsEqual(const Type & type, const Member& member, const void *lhs, const void* rhs) const
{
   if (type.IsPointer()) {
      if (lhs == 0 || rhs == 0)
         return (lhs == 0 && rhs == 0);

      lhs = (const void*)(*(const char**)lhs);
      rhs = (const void*)(*(const char**)rhs);
      return IsEqual(type.ToType(), member, lhs, rhs);
   }
   if (!type.IsFundamental()) {
      Scope scope = type;
      if (scope) {
         bool ret = true;
         for (Member_Iterator iMember = scope.DataMember_Begin();
              iMember != scope.DataMember_End(); ++iMember) {
            size_t offset = iMember->Offset();
            ret &= IsEqual(iMember->TypeOf(), *iMember, ((char*)lhs) + offset, ((char*)rhs) + offset);
         }
         if (!ret)
            if (member)
               cout << "* in " << member.Name(SCOPED) << endl;
         return ret;
      }
   }

   if (type.IsFunction())
     return IsEqual(type.ReturnType(), member, lhs, rhs);

   bool ret = true;
   switch (Tools::FundamentalType(type)) {
   case kCHAR:         ret = *(char*)lhs == *(char*)rhs; break;
   case kSIGNED_CHAR:  ret = *(signed char*)lhs == *(signed char*)rhs; break;
   case kSHORT_INT:    ret = *(short int*)lhs == *(short int*)rhs; break;
   case kINT:          ret = *(int*)lhs == *(int*)rhs; break;
   case kLONG_INT:     ret = *(long int*)lhs == *(long int*)rhs; break;
   case kUNSIGNED_CHAR: ret = *(unsigned char*)lhs == *(unsigned char*)rhs; break;
   case kUNSIGNED_SHORT_INT: ret = *(unsigned short int*)lhs == *(unsigned short int*)rhs; break;
   case kUNSIGNED_INT: ret = *(unsigned int*)lhs == *(unsigned int*)rhs; break;
   case kUNSIGNED_LONG_INT: ret = *(unsigned long int*)lhs == *(unsigned long int*)rhs; break;
   case kBOOL:         ret = *(bool*)lhs == *(bool*)rhs; break;
   case kFLOAT:        ret = *(float*)lhs == *(float*)rhs; break;
   case kDOUBLE:       ret = *(double*)lhs == *(double*)rhs; break;
   case kLONG_DOUBLE:  ret = *(long double*)lhs == *(long double*)rhs; break;
   case kVOID:         ret = *(int*)lhs == *(int*)rhs; break;
   case kLONGLONG:     ret = *(longlong*)lhs == *(longlong*)rhs; break;
   case kULONGLONG:    ret = *(ulonglong*)lhs == *(ulonglong*)rhs; break;
   default:
      cerr << "InspectorBase::IsEqual(): type " << type.Name(SCOPED) << " not handled!" << endl;
      return false;
   };

   if (!ret) {
      if (member) {
         Dump(cout, member, lhs, "* DIFFERENCE detected: ");
         Dump(cout, member, rhs, "*                 and: ");
      } else {
         Dump(cout, type, lhs, "* DIFFERENCE detected: ");
         Dump(cout, type, rhs, "*                 and: ");
      }
   }

   return ret;
}

Inspect::InspectorGenSource::InspectorGenSource(const Scope& scope, const char* header):
   InspectorBase(scope), fHeader(header)
{
   fName = "Inspector_";
   fName += GetScope().Name(SCOPED);
   repl(fName, "::", "__");
   repl(fName, "<", "lT");
   repl(fName, ">", "gT");
   repl(fName, "*", "pT");
   repl(fName, " ", "sP");
   repl(fName, "&", "rE");
}


void Inspect::InspectorGenSource::WriteSource()
{
   std::string scopename(GetScope().Name(SCOPED));
   std::string inspectorName(GetName());
   Type typeVoid = Type::ByName("void");

   ofstream out((inspectorName + ".hh").c_str());
   out << "#include \"" << fHeader << "\""  << endl
       << "#include \"Inspector.hh\"" << endl
       << "using namespace Inspect;" << endl
       << "using namespace ROOT::Reflex;" << endl
       << "class " << inspectorName << ": public InspectorBase {" << endl
       << " private:" << endl
       << "   " << scopename << "* fObj; " << endl
       << " public:" << endl
       << "   " << inspectorName << "(" << scopename << "* obj):" << endl
       << "      InspectorBase(Scope::ByName(\"" << scopename << "\")), "
       << "fObj(obj)" << endl
       << "   {}" << endl
       << "   ~" << inspectorName << "() {}" << endl
       << "   void Inspect() {" << endl
       << "      if (fActual) delete fActual;" << endl
       << "      fActual = new ResultMap_t;" << endl
       << "      ResultMap_t& results = *fActual;" << endl
       << "      Random r;" << endl
       << "      results.clear();" << endl
       << "      Member_Iterator iDM = GetScope().DataMember_Begin();" << endl;

   for (Member_Iterator iDM = GetScope().DataMember_Begin();
        iDM != GetScope().DataMember_End(); ++iDM)
      out << "      results[*iDM] = &fObj->" << iDM->Name() << "; ++iDM;" << endl;
   out << "      assert(iDM == GetScope().DataMember_End());" << endl;

   out << "      Member_Iterator iFM = GetScope().FunctionMember_Begin();" << endl;

   for (Member_Iterator iFM = GetScope().FunctionMember_Begin();
        iFM != GetScope().FunctionMember_End(); ++iFM) {
      Type tFunc(iFM->TypeOf());
      if (tFunc.ReturnType() == typeVoid || iFM->IsArtificial()) {
         out << "      ++iFM; // " << tFunc.ReturnType().Name(SCOPED|QUALIFIED) 
             << " " << iFM->Name(SCOPED | QUALIFIED) << "(";
         StdString_Iterator iParamName = iFM->FunctionParameterName_Begin();
         for (Type_Iterator iParam = tFunc.FunctionParameter_Begin();
              iParam != tFunc.FunctionParameter_End(); ++iParam, ++iParamName) {
            if (iParam != tFunc.FunctionParameter_Begin()) out << ", ";
            out << iParam->Name(SCOPED | QUALIFIED) << " " << *iParamName;
         }
         out << ")" << endl;
         continue;
      }

      unsigned int idx = 0;
      out << "      {" << endl;
      for (Type_Iterator iParam = tFunc.FunctionParameter_Begin();
           iParam != tFunc.FunctionParameter_End(); ++ iParam) {
         if (iParam->IsPointer())
            out << "         " << iParam->Name(SCOPED) << " par" << idx 
                << " = new " << iParam->Name(SCOPED) << ";" << endl;
         else
            out << "         " << iParam->Name(SCOPED) << " par" << idx << ";" << endl;
         if (iParam->ToType().IsFundamental() 
             || iParam->IsPointer() && iParam->ToType().IsFundamental())
         out << "         r(Dummy::Member(), " << (iParam->IsPointer()?"*":"") << "par" << idx << ");" << endl;
      }

      idx = 0;
      out << "         results[*iFM] = new " << tFunc.ReturnType().Name(SCOPED) << "((";
      if (tFunc.IsConst())
         out << "(const " << scopename <<"*)";
      out << "fObj)->" << iFM->Name() << "(";
      for (Type_Iterator iParam = tFunc.FunctionParameter_Begin();
           iParam != tFunc.FunctionParameter_End(); ++ iParam) {
         if (idx) out << ", ";
         out <<"(par" << idx << ")";
      }
      out << ")); ++iFM;" << endl;
      out << "      }" << endl;
   }
   out << "      assert(iFM == GetScope().FunctionMember_End());" << endl;

   out << "   }" << endl
       << "   void InitDataMembers() {" << endl
       << "      Random r;" << endl
       << "      Member_Iterator iDM = GetScope().DataMember_Begin();" << endl;

   for (Member_Iterator iDM = GetScope().DataMember_Begin();
        iDM != GetScope().DataMember_End(); ++ iDM) {
      if (iDM->TypeOf().IsPointer()) {
         out << "      {  " << iDM->TypeOf().Name(SCOPED) << " ptr = new " << iDM->TypeOf().ToType().Name(SCOPED) << ";" << endl;
         if (iDM->TypeOf().ToType().IsFundamental())
            out << "        r(Dummy::Member(), *ptr);" << endl;
         out << "         fObj->" << iDM->Name() << " = ptr;" << endl
             << "      }" << endl;
      } else
         out << "      r(*iDM, fObj->" << iDM->Name() << "); ++iDM;" << endl;
   }
   out << "   }" << endl;

   out << "};" << endl;

   out << "class Register_" << inspectorName << " { " << endl
       << " public:" << endl
       << "   static InspectorBase* InspectorGenerator(void* o) {" << endl
       << "      return new " << inspectorName << "((" << scopename << "*)o); }" << endl
       << "   Register_" << inspectorName << "() {" << endl
       << "      InspectorGenSource::RegisterInspector( "
       << "Scope::ByName(\"" << scopename << "\"), "
       << "InspectorGenerator); }" << endl
       << "};" << endl
       << "static Register_" << inspectorName << " static_Register_" << inspectorName << ";" << endl;
}
