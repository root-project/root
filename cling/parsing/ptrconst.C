class Object {};

class myClass : public Object {
public:
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
	
