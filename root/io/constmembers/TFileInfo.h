#include <TNamed.h>
class TString;

class TFileInfo : public TObject {
public:
  const TString name;
  const TNamed  var1;

  const TString *var2;
  const TNamed  *var3;

  const TString *const var4;
  const TNamed  *const var5;

  TString *const var4a;
  TNamed  *const var5a;


  const TString var6[2];
  const TNamed  var7[3];
  

  //  const Int_t var8[8]; I do not know how to initialize this!
  const Int_t n;
  const Int_t *var9; //[n]
  const Int_t *var10; 
  const Int_t *const var11; //[n]
  const Int_t *const var12;


  const Int_t mode;
  const Int_t size;
  const Int_t mtime;

  TFileInfo();
  TFileInfo(const char *my_name, Int_t my_mode, Int_t my_size,
            Int_t my_mtime):
  name(my_name),
  mode(my_mode),
  size(my_size),
     mtime(my_mtime),
     var4(0),
     var5(0),
     var4a(0),
     var5a(0),
     n(0),
     var11(0),
     var12(0)
  {}
  virtual ~TFileInfo();
  ClassDef (TFileInfo, 1)
};
