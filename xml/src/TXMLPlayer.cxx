#include "TXMLPlayer.h"

#include "Riostream.h"
#include "TClass.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TObjArray.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMethodCall.h"
#include "TFunction.h"

const char* tab1 = "   ";
const char* tab2 = "      ";
const char* tab3 = "         ";
const char* tab4 = "            ";

const char* names_xmlfileclass = "TXmlFile";
const char* names_funcseparator  = "//_______________________________________________________";


ClassImp(TXMLPlayer);


//______________________________________________________________________________
TXMLPlayer::TXMLPlayer() : TObject() 
{
    
}


//______________________________________________________________________________
TXMLPlayer::~TXMLPlayer() 
{
}

//______________________________________________________________________________
TString TXMLPlayer::GetStreamerName(TClass* cl) 
{
  if (cl==0) return "";
  TString res = cl->GetName();
  res += "_streamer";
  return res;
}
      
//______________________________________________________________________________
Int_t TXMLPlayer::ProduceCode(TList* cllist, const char* filename) 
{
   if ((cllist==0) || (filename==0)) return -1;
   
   ofstream fh(TString(filename)+".h");
   ofstream fs(TString(filename)+".cxx");
   
   fh << "// generated header file" << endl << endl;
   fh << "#ifndef " << filename << "_h" << endl;
   fh << "#define " << filename << "_h" << endl << endl;
   
   fh << "#include \"" << names_xmlfileclass << ".h\"" << endl << endl;
   
   fs << "// generated source file" << endl << endl;
   fs << "#include \"" << filename << ".h\"" << endl << endl;
   
   
   // produce class forward declaration and appropriate include
   
   TObjArray inclfiles;
   TIter iter(cllist);
   TClass* cl = 0;
   while ((cl = (TClass*) iter()) != 0) {
//     fh << "class " << cl->GetName() << ";" << endl;
     if (inclfiles.FindObject(cl->GetDeclFileName())==0) {
        fs << "#include \"" << cl->GetDeclFileName() << "\"" << endl;
        inclfiles.Add(new TNamed(cl->GetDeclFileName(),""));
     }
   }
   inclfiles.Delete();
   
   fh << endl;
   fs << endl;
   
   
   // produce streamers declarations and implementations
   
   iter.Reset();
   
   while ((cl = (TClass*) iter()) != 0) {
       
      fh << "extern void* " << GetStreamerName(cl) << "(" 
         << names_xmlfileclass << " &buf, void* ptr = 0, bool checktypes = true);" << endl << endl; 
      
      ProduceStreamerSource(fs, cl, cllist);
   }

   fh << "#endif" << endl << endl;
   fs << endl << endl;
    
   return 0; 
}

//______________________________________________________________________________
TString TXMLPlayer::GetMemberTypeName(TDataMember* member)
{
   if (member==0) return "int"; 
    
   if (member->IsBasic())
   switch (member->GetDataType()->GetType()) { 
     case kChar_t:     return "char";
     case kShort_t:    return "short";
     case kInt_t:      return "int";
     case kLong_t:     return "long";
     case kLong64_t:   return "long long";
     case kFloat_t:    return "float";
     case kDouble32_t:
     case kDouble_t:   return "double";
     case kUChar_t:    {
       char first = member->GetDataType()->GetTypeName()[0];  
       if ((first=='B') || (first=='b')) return "bool";
                                    else return "unsigned char";         
     }
     case kUShort_t:   return "unsigned short";
     case kUInt_t:     return "unsigned int";
     case kULong_t:    return "unsigned long";
     case kULong64_t:  return "unsigned long long";
   }
   
   if (member->IsEnum()) return "int";
   
   return member->GetTypeName();
}

//______________________________________________________________________________
TString TXMLPlayer::GetBasicTypeName(TStreamerElement* el)
{
   if (el->GetType() == TStreamerInfo::kCounter) return "int"; 

   switch (el->GetType() % 20) {
     case TStreamerInfo::kChar:     return "char";
     case TStreamerInfo::kShort:    return "short";
     case TStreamerInfo::kInt:      return "int";
     case TStreamerInfo::kLong:     return "long";
     case TStreamerInfo::kLong64:   return "long long";
     case TStreamerInfo::kFloat:    return "float";
     case TStreamerInfo::kDouble32:
     case TStreamerInfo::kDouble:   return "double";
     case TStreamerInfo::kUChar: {
       char first = el->GetTypeNameBasic()[0];  
       if ((first=='B') || (first=='b')) return "bool";
                                    else return "unsigned char";         
     }
     case TStreamerInfo::kUShort:   return "unsigned short";
     case TStreamerInfo::kUInt:     return "unsigned int";
     case TStreamerInfo::kULong:    return "unsigned long";
     case TStreamerInfo::kULong64:  return "unsigned long long";
   }
   return "int";  
}

//______________________________________________________________________________
TString TXMLPlayer::GetBasicTypeReaderMethodName(TStreamerElement* el) 
{
   if (el->GetType() == TStreamerInfo::kCounter) return "ReadInt"; 
    
   switch (el->GetType() % 20) {
     case TStreamerInfo::kChar:     return "ReadChar";
     case TStreamerInfo::kShort:    return "ReadShort";
     case TStreamerInfo::kInt:      return "ReadInt";
     case TStreamerInfo::kLong:     return "ReadLong";
     case TStreamerInfo::kLong64:   return "ReadLong64";
     case TStreamerInfo::kFloat:    return "ReadFloat";
     case TStreamerInfo::kDouble32:
     case TStreamerInfo::kDouble:   return "ReadDouble";
     case TStreamerInfo::kUChar: {
       char first = el->GetTypeNameBasic()[0];  
       if ((first=='B') || (first=='b')) return "ReadBool";
                                    else return "ReadUChar";         
     }
     case TStreamerInfo::kUShort:   return "ReadUShort";
     case TStreamerInfo::kUInt:     return "ReadUInt";
     case TStreamerInfo::kULong:    return "ReadULong";
     case TStreamerInfo::kULong64:  return "ReadULong64";
   }
   return "ReadValue";  
}

//______________________________________________________________________________
const char* TXMLPlayer::ElementGetter(TClass* cl, const char* membername, int specials)
// specials = 0 - do nothing
//            1 - cast to data type
//            2 - produce pointer on given member 
//            3 - skip casting when produce pointer by buf.P() function
{
   TClass* membercl = cl ? cl->GetBaseDataMember(membername) : 0;
   TDataMember* member = membercl ? membercl->GetDataMember(membername) : 0;
   TMethodCall* mgetter = member ? member->GetterMethod(cl) : 0;
   
   if ((mgetter!=0) && (mgetter->GetMethod()->Property() & kIsPublic)) {
      fGetterName = "obj->";
      fGetterName += mgetter->GetMethodName();
      fGetterName += "()";
   } else 
   if ((member==0) || ((member->Property() & kIsPublic) != 0)) {
      fGetterName = "obj->";
      fGetterName += membername;
   } else {
      fGetterName = "";
      bool deref = (member->GetArrayDim()==0) && (specials!=2);
      if (deref) fGetterName += "*(";
      if (specials!=3) {
         fGetterName += "("; 
         if (member->Property() & kIsConstant) fGetterName += "const ";
         fGetterName += GetMemberTypeName(member);
         if (member->IsaPointer()) fGetterName+="*";
         fGetterName += "*) ";
      }
      fGetterName += "buf.P(obj,";
      fGetterName += member->GetOffset();
      fGetterName += ")";
      if (deref) fGetterName += ")";
      specials = 0;
   }
   
   if ((specials==1) && (member!=0)) {
      TString cast = "(";
      cast += GetMemberTypeName(member);   
      if (member->IsaPointer() || (member->GetArrayDim()>0)) cast += "*";
      cast += ") ";
      cast += fGetterName;
      fGetterName = cast;
   }
   
   if ((specials==2) && (member!=0)) {
      TString buf = "&(";
      buf += fGetterName; 
      buf += ")";
      fGetterName = buf;
   }
   
   return fGetterName.Data();
}

//______________________________________________________________________________
const char* TXMLPlayer::ElementSetter(TClass* cl, const char* membername, char* endch)
{
   strcpy(endch,""); 
   
   TClass* membercl = cl ? cl->GetBaseDataMember(membername) : 0;
   TDataMember* member = membercl ? membercl->GetDataMember(membername) : 0;
   TMethodCall* msetter = member ? member->SetterMethod(cl) : 0;
   
   if ((msetter!=0) && (msetter->GetMethod()->Property() & kIsPublic)) {
      fSetterName = "obj->";
      fSetterName += msetter->GetMethodName();
      fSetterName += "(";
      strcpy(endch,")"); 
   } else 
   if ((member==0) || (member->Property() & kIsPublic) != 0) {
      fSetterName = "obj->";
      fSetterName += membername;
      fSetterName += " = ";
   } else {
      fSetterName = "";
      if (member->GetArrayDim()==0) fSetterName += "*";
      fSetterName += "((";
      if (member->Property() & kIsConstant) fSetterName += "const ";
      fSetterName += GetMemberTypeName(member);
      if (member->IsaPointer()) fSetterName += "*";
      fSetterName += "*) buf.P(obj,";
      fSetterName += member->GetOffset();
      fSetterName += ")) = ";
   }
   return fSetterName.Data();
}

//______________________________________________________________________________
TString TXMLPlayer::GetElemName(TStreamerElement* el)
{
  TString res = "obj->";
  res += el->GetName();
  return res; 
}

//______________________________________________________________________________
void TXMLPlayer::ProduceStreamerSource(ostream& fs, TClass* cl, TList* cllist) 
{
   if (cl==0) return; 
   TStreamerInfo* info = cl->GetStreamerInfo();
   TObjArray* elements = info->GetElements();
   if (elements==0) return;

   fs << names_funcseparator << endl;
   fs << "void* " << GetStreamerName(cl) << "(" 
         << names_xmlfileclass << " &buf, void* ptr, bool checktypes)" << endl; 
   fs << "{" << endl;
   fs << tab1 << cl->GetName() << " *obj = (" << cl->GetName() << "*) ptr;" << endl;
   
   fs << tab1 << "if (buf.IsReading()) { " << endl;
   
   TIter iter(cllist);
   TClass* c1 = 0;
   bool firstchild = true;
   
   while ((c1 = (TClass*) iter()) != 0) {
      if (c1==cl) continue;
      if (c1->GetListOfBases()->FindObject(cl->GetName())==0) continue;
      if (firstchild) {
         fs << tab2 << "if (checktypes) {" << endl;
         fs << tab3 << "void* ";
         firstchild = false;
      } else 
         fs << tab3;
      fs << "res = " << GetStreamerName(c1) 
         << "(buf, dynamic_cast<" << c1->GetName() << "*>(obj));" << endl;
      fs << tab3 << "if (res) return dynamic_cast<" << cl->GetName() 
         << "*>(("<< c1->GetName() << " *) res);" << endl;
   }
   if (!firstchild) fs << tab2 << "}" << endl;
   
   fs << tab2 << "if (!buf.CheckClassNode(\"" << cl->GetName() << "\", " 
              << info->GetClassVersion() << ")) return 0;" << endl;
              
   fs << tab2 << "if (obj==0) obj = new " << cl->GetName() << ";" << endl;
   
   for (int n=0;n<=elements->GetLast();n++) {
       
      TStreamerElement* el = dynamic_cast<TStreamerElement*> (elements->At(n));
      if (el==0) continue; 
      
      Int_t typ = el->GetType();
      
      switch (typ) {
         // basic types
         case TStreamerInfo::kChar:              
         case TStreamerInfo::kShort:
         case TStreamerInfo::kInt:  
         case TStreamerInfo::kLong:  
         case TStreamerInfo::kLong64:
         case TStreamerInfo::kFloat: 
         case TStreamerInfo::kDouble:
         case TStreamerInfo::kUChar:             
         case TStreamerInfo::kUShort:
         case TStreamerInfo::kUInt:  
         case TStreamerInfo::kULong: 
         case TStreamerInfo::kULong64: 
         case TStreamerInfo::kDouble32:
         case TStreamerInfo::kCounter: {
            char endch[5]; 
            fs << tab2 << ElementSetter(cl, el->GetName(), endch);
            fs << "buf." << GetBasicTypeReaderMethodName(el) 
               << "(\"" << el->GetName() << "\")" << endch << ";" << endl;
            continue;
         }
         
         // array of basic types like bool[10]
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:     
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64: 
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32: {
            fs << tab2 << "buf.ReadArray("
                       << ElementGetter(cl, el->GetName(), (el->GetArrayDim()>1) ? 1 : 0);
            fs         << ", " << el->GetArrayLength()
                       << ", \"" << el->GetName() << "\");" << endl; 
            continue;   
         }
         
         // array of basic types like bool[n] 
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64:
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32: {
            TStreamerBasicPointer* elp = dynamic_cast<TStreamerBasicPointer*> (el);
            if (elp==0) {
              cout << "fatal error with TStreamerBasicPointer" << endl;  
              continue;
            }
            char endch[5]; 
            
            fs << tab2 << ElementSetter(cl, el->GetName(), endch);
            fs         << "buf.ReadArray(" << ElementGetter(cl, el->GetName());
            fs         << ", " << ElementGetter(cl, elp->GetCountName());
            fs         << ", \"" << el->GetName() << "\", true)" << endch << ";" << endl; 
            continue;   
         }
         
         case TStreamerInfo::kCharStar: {
            char endch[5]; 
            fs << tab2 << ElementSetter(cl, el->GetName(), endch);
            fs         << "buf.ReadCharStar(" << ElementGetter(cl, el->GetName());
            fs         << ", \"" << el->GetName() << "\")" << endch << ";" << endl; 
            continue;   
         }
         
         case TStreamerInfo::kBase: {
            fs << tab2 << GetStreamerName(el->GetClassPointer()) 
               << "(buf, dynamic_cast<" << el->GetClassPointer()->GetName() 
               << "*>(obj), false);" << endl;
            continue;
         }
         
         // Class*   Class not derived from TObject and with comment field //->
         case TStreamerInfo::kAnyp:     
         case TStreamerInfo::kAnyp    + TStreamerInfo::kOffsetL: {
            if (el->GetArrayLength()>0) {
               fs << tab2 << "buf.ReadObjectArr(" << ElementGetter(cl, el->GetName());
               fs         << ", " << el->GetArrayLength() << ", -1"
                          << ", \"" << el->GetName() << "\", "
                          << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            } else {
              fs << tab2 << "buf.ReadObject(" << ElementGetter(cl, el->GetName());
              fs         << ", \"" << el->GetName() << "\", "
                         << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            }
            continue;
         }
         
         // Class*   Class not derived from TObject and no comment
         case TStreamerInfo::kAnyP: 
         case TStreamerInfo::kAnyP + TStreamerInfo::kOffsetL: {
            if (el->GetArrayLength()>0) {
              fs << tab2 << "for (int n=0;n<" << el->GetArrayLength() << ";n++) "
                         << "delete (" << ElementGetter(cl, el->GetName()) << ")[n];" << endl;
              fs << tab2 << "buf.ReadObjectPtrArr((void**) " << ElementGetter(cl, el->GetName(), 3);
              fs         << ", " << el->GetArrayLength() 
                         << ", \"" << el->GetName() << "\", "
                         << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            } else {
              char endch[5]; 
            
              fs << tab2 << "delete " << ElementGetter(cl, el->GetName()) << ";" << endl;
              fs << tab2 << ElementSetter(cl, el->GetName(), endch);
              fs         << "(" << el->GetClassPointer()->GetName() 
                         << "*) buf.ReadObjectPtr(\"" << el->GetName() << "\", "
                         << GetStreamerName(el->GetClassPointer()) 
                         << ")" <<endch << ";" << endl;
            }
            continue;
         }
         
         
/*         
         // Class*   Class not derived from TObject and no virtual table and no comment   
         case TStreamerInfo::kAnyPnoVT:     
         case TStreamerInfo::kAnyPnoVT + TStreamerInfo::kOffsetL: {
            fs << tab2 << "// read object AnyPnoVT pointer " << el->GetName() << endl;  
            continue;
         }
         
         // Pointer to container with no virtual table (stl) and no comment   
         case TStreamerInfo::kSTLp:                
         // array of pointers to container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL: { 
            fs << tab2 << "// read STL container object pointer " << el->GetName() << endl;  
            continue;
         }
         
         // container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTL:             
         // array of containers with no virtual table (stl) and no comment
         case TStreamerInfo::kSTL + TStreamerInfo::kOffsetL: {
            fs << tab2 << "// read STL container object " << el->GetName() << endl;  
            continue;
         } 
*/         
         case TStreamerInfo::kAny: {  // Class  NOT derived from TObject
            fs << tab2 << "buf.ReadObject(" << ElementGetter(cl, el->GetName(), 2);
            fs         << ", \"" << el->GetName() << "\", "
                       << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            continue;
         }
       
         case TStreamerInfo::kAny + TStreamerInfo::kOffsetL: { // Class  NOT derived from TObject[8]
            fs << tab2 << "buf.ReadObjectArr(" << ElementGetter(cl, el->GetName());
            fs         << ", " << el->GetArrayLength() 
                       << ", sizeof(" << el->GetClassPointer()->GetName()
                       << "), \"" << el->GetName() << "\", "
                       << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            continue;
         }
         
         default:
           fs << tab2 << "buf.SkipMember(\"" << el->GetName() 
                      << "\");   // sinfo type " << el->GetType() 
                      << " not supported" << endl;

      }
   }
   
   fs << tab2 << "buf.EndClassNode();" << endl;
   
   fs << tab1 << "} else {" << endl;
   
   // generation of writing part of class streamer
      
   fs << tab2 << "if (obj==0) return 0;" << endl;
   
   firstchild = true;
   iter.Reset();
   while ((c1 = (TClass*) iter()) != 0) {
      if (c1==cl) continue;
      if (c1->GetListOfBases()->FindObject(cl->GetName())==0) continue;
      if (firstchild) {
        firstchild = false;
        fs << tab2 << "if (checktypes) {" << endl;  
      }
      fs << tab3 << "if (dynamic_cast<" << c1->GetName() << "*>(obj))" << endl;
      fs << tab4 << "return " << GetStreamerName(c1) << "(buf, dynamic_cast<" << c1->GetName() << "*>(obj));" << endl;
   }
   if (!firstchild) fs << tab2 << "}" << endl;
   
   fs << tab2 << "buf.StartClassNode(\"" << cl->GetName() << "\", " 
              << info->GetClassVersion() << ");" << endl;
   
   for (int n=0;n<=elements->GetLast();n++) {
       
      TStreamerElement* el = dynamic_cast<TStreamerElement*> (elements->At(n));
      if (el==0) continue; 
      
      Int_t typ = el->GetType();
      
      switch (typ) {
         // write basic types
         case TStreamerInfo::kChar:              
         case TStreamerInfo::kShort:
         case TStreamerInfo::kInt:  
         case TStreamerInfo::kLong:  
         case TStreamerInfo::kLong64:
         case TStreamerInfo::kFloat: 
         case TStreamerInfo::kDouble:
         case TStreamerInfo::kUChar:             
         case TStreamerInfo::kUShort:
         case TStreamerInfo::kUInt:  
         case TStreamerInfo::kULong: 
         case TStreamerInfo::kULong64: 
         case TStreamerInfo::kDouble32: 
         case TStreamerInfo::kCounter: {
            fs << tab2 << "buf.WriteValue(" << ElementGetter(cl, el->GetName())
                       << ", \"" << el->GetName() << "\");" << endl; 
            continue;
         }
         
         // array of basic types   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kChar:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kShort:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kInt:     
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kLong64:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kFloat:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUChar:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUShort:  
         case TStreamerInfo::kOffsetL + TStreamerInfo::kUInt:    
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong:   
         case TStreamerInfo::kOffsetL + TStreamerInfo::kULong64: 
         case TStreamerInfo::kOffsetL + TStreamerInfo::kDouble32: {
            fs << tab2 << "buf.WriteArray("
                       << ElementGetter(cl, el->GetName(), (el->GetArrayDim()>1) ? 1 : 0);
            fs         << ", " << el->GetArrayLength()
                       << ", \"" << el->GetName() << "\");" << endl; 
            continue;   
         }
         
         case TStreamerInfo::kOffsetP + TStreamerInfo::kChar:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kShort:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kInt:     
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kLong64:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kFloat:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUChar:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUShort:  
         case TStreamerInfo::kOffsetP + TStreamerInfo::kUInt:    
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong:   
         case TStreamerInfo::kOffsetP + TStreamerInfo::kULong64:
         case TStreamerInfo::kOffsetP + TStreamerInfo::kDouble32: {
            TStreamerBasicPointer* elp = dynamic_cast<TStreamerBasicPointer*> (el);
            if (elp==0) {
              cout << "fatal error with TStreamerBasicPointer" << endl;  
              continue;
            }
            fs << tab2 << "buf.WriteArray(" << ElementGetter(cl, el->GetName());
            fs         << ", " << ElementGetter(cl, elp->GetCountName())
                       << ", \"" << el->GetName() << "\", true);" << endl; 
            continue;   
         }
         
         case TStreamerInfo::kCharStar: {
            fs << tab2 << "buf.WriteCharStar(" << ElementGetter(cl, el->GetName())
                       << ", \"" << el->GetName() << "\");" << endl; 
            continue;   
         }
         
         case TStreamerInfo::kBase: {
            fs << tab2 << GetStreamerName(el->GetClassPointer()) 
               << "(buf, dynamic_cast<" << el->GetClassPointer()->GetName() 
               << "*>(obj), false);" << endl;
            continue;
         }
         
         // Class*   Class not derived from TObject and with comment field //->
         case TStreamerInfo::kAnyp:     
         case TStreamerInfo::kAnyp    + TStreamerInfo::kOffsetL: {
            if (el->GetArrayLength()>0) {
               fs << tab2 << "buf.WriteObjectArr(" << ElementGetter(cl, el->GetName());
               fs         << ", " << el->GetArrayLength() << ", -1"
                          << ", \"" << el->GetName() << "\", "
                          << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            } else {
               fs << tab2 << "buf.WriteObject(" << ElementGetter(cl, el->GetName());
               fs         << ", \"" << el->GetName() << "\", "
                          << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            }
            continue;
         }
         
         // Class*   Class not derived from TObject and no comment
         case TStreamerInfo::kAnyP: 
         case TStreamerInfo::kAnyP + TStreamerInfo::kOffsetL: {
            if (el->GetArrayLength()>0) {    
              fs << tab2 << "buf.WriteObjectPtrArr((void**) " << ElementGetter(cl, el->GetName(), 3);
              fs         << ", " << el->GetArrayLength() 
                         << ", \"" << el->GetName() << "\", "
                         << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            } else {
              fs << tab2 << "buf.WriteObjectPtr(" << ElementGetter(cl, el->GetName());
              fs         << ", \"" << el->GetName() << "\", "
                         << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            }
            continue;
         }
/*         
         // Class*   Class not derived from TObject and no virtual table and no comment   
         case TStreamerInfo::kAnyPnoVT:     
         case TStreamerInfo::kAnyPnoVT + TStreamerInfo::kOffsetL: {
            fs << tab2 << "// write object AnyPnoVT pointer " << el->GetName() << endl;  
            continue;
         }
         
         // Pointer to container with no virtual table (stl) and no comment   
         case TStreamerInfo::kSTLp:                
         // array of pointers to container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTLp + TStreamerInfo::kOffsetL: { 
            fs << tab2 << "// write STL container object pointer " << el->GetName() << endl;  
            continue;
         }
         
         // container with no virtual table (stl) and no comment
         case TStreamerInfo::kSTL:             
         // array of containers with no virtual table (stl) and no comment
         case TStreamerInfo::kSTL + TStreamerInfo::kOffsetL: {
            fs << tab2 << "// write STL container object " << el->GetName() << endl;  
            continue;
         } 
*/         
         case TStreamerInfo::kAny: {    // Class  NOT derived from TObject
            fs << tab2 << "buf.WriteObject(" << ElementGetter(cl, el->GetName(), 2);
            fs         << ", \"" << el->GetName() << "\", "
                       << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            continue;
         }
        
         case TStreamerInfo::kAny    + TStreamerInfo::kOffsetL: {
            fs << tab2 << "buf.WriteObjectArr(" << ElementGetter(cl, el->GetName());
            fs         << ", " << el->GetArrayLength() 
                       << ", sizeof(" << el->GetClassPointer()->GetName()
                       << "), \"" << el->GetName() << "\", "
                       << GetStreamerName(el->GetClassPointer()) << ");" << endl;
            continue;
         }
         
         default:
           fs << tab2 << "buf.MakeEmptyMember(\"" << el->GetName() 
                      << "\");   // sinfo type " << el->GetType() 
                      << " not supported" << endl;
      }
   }
   
   fs << tab2 << "buf.EndClassNode();" << endl;
   
   fs << tab1 << "}" << endl;
   fs << tab1 << "return obj;" << endl;
   fs << "}" << endl << endl;
}
    