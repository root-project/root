#include "TShape.h"
#include <iostream.h>


class Simple : public TObject {

private:
   Int_t fID; // id number
   TShape* fShape; // pointer to base class shape

public:

   Simple() : fID(0), fShape(0) { }
   Simple(Int_t id, TShape* shape): fID(id), fShape(shape) { }
   virtual ~Simple();
   virtual void Print(Option_t *option = "") const;

   ClassDef(Simple,1)  //Simple class
   private: static TClass *fgIsA; public: static TClass *Class(); static const char *Class_Name(); static Version_t Class_Version() { return 1; } static void Dictionary(); virtual TClass *IsA() const { return Simple::Class(); } virtual void ShowMembers(TMemberInspector &insp, char *parent); virtual void Streamer(TBuffer &b); void StreamerNVirtual(TBuffer &b) { Simple::Streamer(b); } static const char *DeclFileName() { return "Simple.h"; } static int DeclFileLine() { return 23; } static const char *ImplFileName(); static int ImplFileLine(); //Simple class
};

ClassImp(Simple)
ClassImp(Simple)
ClassImp(Simple)

Simple::~Simple() {
// Destructor
  if (fShape) {
    delete fShape;
    fShape =0;
  }
}

void Simple::Print(Option_t *option) const {
  // Print the contents
  cout << "fID= " << fID << endl;
  fShape -> Print();

}
