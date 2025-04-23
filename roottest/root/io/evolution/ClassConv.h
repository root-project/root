#if VERSION==2
namespace MyLib {
#elif VERSION==3
namespace OtherLib {
#endif
   class Typedefed {
   public:
      int var;
   };
   class Inside {
      int i;
   public:
      Inside(int input) : i(input) {};
      int GetValue() { return i; }
   };
#if VERSION==2 || VERSION==3
} // namespace MyLib
#endif

#if VERSION==2
typedef MyLib::Typedefed Typedefed;
#elif VERSION==3
typedef OtherLib::Typedefed Typedefed;
#endif

class TopLevel {
public:
#if VERSION==2
   MyLib::Inside in;
#elif VERSION==3
   OtherLib::Inside in;
#else
   Inside in;
#endif
#if VERSION==2
   MyLib::Typedefed t;
#else
   Typedefed t;
#endif
public:
   TopLevel(int input = 99) : in(input) {}
   int GetValue() { return in.GetValue(); }
};
