//namespace L {
   // template <typename T> class AA;
   // template <typename T> class AA {};
   // class AB: public virtual AA<char> {};

   // class Redecl;
   // class RedeclImp { Redecl* p; };

   // class Redecl { public: int M; };

   // class Redecl;

   // class RedeclUse { Redecl* use; };

   // class Redecl;

   //template <typename T> int fwddeclThenDefinedInLeftRight(T);

   //int usesfwddeclThenDefinedInLeftRight() {
   //  return fwddeclThenDefinedInLeftRight((char)1); 
   //}

   // template <typename T> int fwddeclThenDefinedInLeftRight(T) { return 0; }

   template <typename T> int definedInLeft(T) { return 1; }
   //int usesdefinedInLeft() { return definedInLeft((int)0); }

   // template <typename T> int definedInLeftSpecializedInRight(T) { return 2; }
   // int usesdefinedInLeftSpecializedInRightInt() {
   //    return definedInLeftSpecializedInRight((int)0); }
  //}
