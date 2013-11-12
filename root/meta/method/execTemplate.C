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

void execTemplate()
{
   Check(TClass::GetClass("UserClass"));
   Check(TClass::GetClass("UserSpace"));
}