#include <type_traits>
#include <utility>

class UserClass
{
public:
   template <typename T> T getA() { return 1; }
   template <typename T, typename Q> T getB() { return 1; }

   template <typename T> void setA(T) {  }
   template <typename T, typename Q> void setB(T) {  }

};

namespace UserSpace
{
   template <typename T> T getA() { return 1; }
   template <typename T, typename Q> T getB() { return 1; }

   template <typename T> void setA(T) {  }
   template <typename T, typename Q> void setB(T) {  }
}

template <typename T> void globalSet(T&) {}

template <class T>
struct Outer {
   template <class A, class B> using EnableIfSame_t
      = typename std::enable_if<std::is_same<A, B>::value>::type;

   // Test for methods with all defaulted params, typical for enable-if'ed
   // methods, that TClass::GetListOfMethods() should really report.
   template <class A = T, class = EnableIfSame_t<A, T>>
   Outer() {}

   template <class A = T, class = EnableIfSame_t<A, T>>
   void AFunction(float);

   template <class A = T, class = EnableIfSame_t<A, T>>
   void AFunction(A);

   template <class A = T, class = EnableIfSame_t<A, long>>
   void ThisOneShouldBeDisabled(A);

   void AFunction(double);
};


// ROOT-8422
template<typename T>
struct Wrapper {
   template <typename U = T,
             class = typename std::enable_if<!std::is_reference<U>::value>::type>
  Wrapper(): data() {}

  template<typename V>
  Wrapper(V&& value): data(std::forward<V>(value)) {}

  T data;
};

using WIntRef = Wrapper<int&>;


void CheckTemplate(TClass *cl)
{
   fprintf(stdout,"Looking for function template in %s.\n",cl->GetName());

   TObject *obj;

   obj = cl->GetListOfMethods(false)->FindObject("setB");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find setB without an instantiation.\n");
   }

   obj = cl->GetFunctionTemplate("setB");
   if (obj) {
      fprintf(stdout,"Found %s function template while searching for setB.\n", obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find a function template for setB.\n");
   }

}

void CheckGlobalTemplate()
{
   fprintf(stdout,"Looking fo function template in %s.\n","global scope");

   TObject *obj;

   obj = gROOT->GetListOfGlobalFunctions(false)->FindObject("globalSet");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find globalSet without an instantiation.\n");
   }

   obj = gROOT->GetFunctionTemplate("globalSet");
   if (obj) {
      fprintf(stdout,"Found %s function template while searching for globalSet.\n", obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find a function template for globalSet.\n");
   }

   TClass *cl = TClass::GetClass("TMath");
   obj = cl->GetFunctionTemplate("Mean");
   if (obj) {
      fprintf(stdout,"Found %s function template while searching for TMath::Mean.\n", obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find a function template for  TMath::Mean.\n");
   }
   obj = cl->GetFunctionTemplate("Maen");
   if (obj) {
      fprintf(stdout,"Error: found %s function template while searching for TMath::Maen.\n", obj->GetName());
   } else {
      fprintf(stdout,"As expected did not find a function template for  TMath::Maen.\n");
   }
   obj = cl->GetListOfMethods(true)->FindObject("Mean");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find TMath::Mean without an instantiation.\n");
   }
   obj = cl->GetListOfMethods(true)->FindObject("Maen");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not exist.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find TMath::Maen which does not exist.\n");
   }
   obj = cl->GetMethodAny("Mean");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find TMath::Mean without an instantiation.\n");
   }
   obj = cl->GetMethodAny("Maen");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not exist.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find TMath::Maen which does not exist.\n");
   }

}


void Check(TClass *cl)
{
   fprintf(stdout,"Looking at %s.\n",cl->GetName());

   cl->GetListOfMethods(false)->ls("noaddr");

   TObject *obj;

   obj = cl->GetListOfMethods(false)->FindObject("getA");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find getA without an instantiation.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getA<int>");
   if (obj) {
      fprintf(stdout,"Found %s while searching for getA<int>.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find getA<int>.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getA");
   if (obj) {
      fprintf(stdout,"Found %s while searching for getA.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find getA after getA<int>'s instantiation.\n");
   }

   obj = cl->GetListOfMethods(false)->FindObject("setA");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find setA without an instantiation.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("setA<int>");
   if (obj) {
      fprintf(stdout,"Found %s while searching for setA<int>.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find setA<int>.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("setA");
   if (obj) {
      fprintf(stdout,"Found %s while searching for setA.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find setA after setA<int>'s instantiation.\n");
   }

   obj = cl->GetListOfMethods(false)->FindObject("getB");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not be instantiated.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find getB without an instantiation.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getB<int>");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not exist.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find getB<int>.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getB<int,float>");
   if (obj) {
      fprintf(stdout,"Found %s while searching for getB<int,float>.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find getB<int,float>.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getB<int>");
   if (obj) {
      fprintf(stdout,"Error: Found %s even-though it should not exist.\n",obj->GetName());
   } else {
      fprintf(stdout,"As expected, can not find getB<int>.\n");
   }
   obj = cl->GetListOfMethods(false)->FindObject("getB");
   if (obj) {
      fprintf(stdout,"Found %s while searching for getB.\n",obj->GetName());
   } else {
      fprintf(stdout,"Error: did not find getB after getB<int,float>'s instantiation.\n");
   }


   cl->GetListOfMethods(false)->ls("noaddr");

   fprintf(stdout,"\n");
}


void CheckEnableIf() {
   TClass* cl = TClass::GetClass("Outer<int>");
   cl->GetListOfMethods(false)->ls("noaddr");
   cl->GetListOfMethods(true)->ls("noaddr");
   printf("TClass::New() on Outer<int> does%s create an object\n",
          cl->New() ? "": " not");

   TClass* cl2 = TClass::GetClass("WIntRef");
   cl2->GetListOfMethods()->ls("noaddr");
}

void execTemplate()
{
   Check(TClass::GetClass("UserClass"));
   Check(TClass::GetClass("UserSpace"));
   CheckTemplate(TClass::GetClass("UserClass"));
   CheckTemplate(TClass::GetClass("UserSpace"));

   CheckGlobalTemplate();

   CheckEnableIf();
}
