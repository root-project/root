#if VERSION==2
namespace MyLib {
#endif
   class Inside {
      int i;
   public:
      Inside(int input) : i(input) {};
      int GetValue() { return i; }
   };
#if VERSION==2
} // namespace MyLib
#endif

class TopLevel {
#if VERSION==2
   MyLib::Inside in;
#else
   Inside in;
#endif
public:
   TopLevel(int input = 99) : in(input) {}
   int GetValue() { return in.GetValue(); }
};
