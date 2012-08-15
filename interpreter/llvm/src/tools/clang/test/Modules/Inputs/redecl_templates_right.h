//namespace L {
   /*
   template <typename T> class AA;
   template <typename T> class AA {};
   class AB: public virtual AA<char> {};
   */

  //template <typename T> int fwddeclThenDefinedInLeftRight(T);

  // int usesfwddeclThenDefinedInLeftRightToo() {
  //   return fwddeclThenDefinedInLeftRight((float)1); 
  // }

  // template <typename T> int fwddeclThenDefinedInLeftRight(T) { return 0; }

  template <typename T> int definedInLeft(T);

  // template <typename T> int definedInLeftSpecializedInRight(T);
  // template <> int definedInLeftSpecializedInRight<char>(char) { return 1; }
  // int usesdefinedInLeftSpecializedInRightChar() {
  //   return definedInLeftSpecializedInRight((char)0); 
  // }
//}
