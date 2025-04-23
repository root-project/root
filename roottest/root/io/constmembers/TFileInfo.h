#include <string>
#include <TNamed.h>
class TString;
using std::string;

class TFileInfo : public TObject {
public:
  const TString name;
#ifndef OLD_IO
  // string is NOT support in old I/O
  const string namestl;
#endif
  const TNamed  var1;

  const TString *var2;
  const TNamed  *var3;
#ifndef OLD_IO
  const string  *var3a; // missing wrapper function
  const string  *const var3b; //missing wrapper function
#endif

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

#ifndef OLD_IO
  // the old io does not support references.
  const TString &ref1;
  TString &ref2;
  string  &ref3;
  const string  &ref4;
  const TNamed  &ref5;
  TNamed  &ref6;
#endif

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
#ifndef OLD_IO
     var3b(new string("var3b")),
#endif
     var4(0),
     var5(0),
     var4a(0),
     var5a(0),
     n(0),
     var11(0),
     var12(0)
#ifndef OLD_IO
     ,
     ref1("test01"),
     ref2(*new TString("test02")),
     ref3(*new string("test02")),
     ref4("test04"),
     ref5(TNamed("test05","t05")),
     ref6(*new TNamed("test05","t05"))
#endif
  {}
  virtual ~TFileInfo();
  ClassDef (TFileInfo, 1)
};

class FileInfo {
public:
  const TString name;
  const string namestl; 
  const TNamed  var1;

  const TString *var2;
  const TNamed  *var3;
  const string  *var3a;
  const string  *const var3b;

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

  const TString  &ref1; //
  TString &ref2; //
  string  &ref3;
  const string  &ref4;
  const TNamed  &ref5;
  TNamed  &ref6;

  const Int_t mode;
  const Int_t size;
  const Int_t mtime;

  FileInfo();
  FileInfo(const char *my_name, Int_t my_mode, Int_t my_size,
            Int_t my_mtime):
  name(my_name),
  mode(my_mode),
  size(my_size),
     mtime(my_mtime),
     var3b(new string("var3b")),
     var4(0),
     var5(0),
     var4a(0),
     var5a(0),
     n(0),
     var11(0),
     var12(0),
     ref1("test01"),
     ref2(*new TString("test02")),
     ref3(*new string("test02")),
     ref4("test04"),
     ref5(TNamed("test05","t05")),
     ref6(*new TNamed("test05","t05"))
  {}
  ~FileInfo();
  
};
