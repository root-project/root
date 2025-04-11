class Object {};

class myClass : public Object {
public:
  myClass(): mTo{0} {}
   Object * const &front() const {return mTo;}
   Object *mTo;
};
#ifdef __MAKECINT__
#pragma link C++ class myClass;
#endif

int ptrconst()
{
   myClass m;
   return m.mTo != m.front();
}
