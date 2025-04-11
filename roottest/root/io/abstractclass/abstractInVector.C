#include <vector>
#include <TObject.h>
#include <string>
using namespace std;

class SimuModule {
public:

    SimuModule( const string& name) : fName(name) {};
    virtual ~SimuModule() {};

    inline const string& GetName() const { return fName; }
    virtual bool Init() = 0;
    virtual bool PreProcess() = 0;
    virtual bool Process() = 0;
    virtual bool PostProcess() = 0;
    virtual bool Done() = 0;
    virtual void UserMemoryClean() {}

public:
   //EusoConfigClass(Simu,SimuModule)

private:
    string fName;

   ClassDef(SimuModule,1);
};

class Module : public SimuModule {
public:
    Module() : SimuModule("default") {};
    Module( const string& name ) : SimuModule(name){};
    virtual ~Module() {};

    virtual bool Init() {return true;};
    virtual bool PreProcess() {return true;};
    virtual bool Process() {return true;};
    virtual bool PostProcess() {return true;};
    virtual bool Done() {return true;};
    virtual void UserMemoryClean() {}

public:
   // EusoConfigClass(Simu,SimuModule)

private:
   ClassDef(Module,1);
};



class SimuModuleFactory {
public:
    // ctor
    SimuModuleFactory(const string&) {};

    // dtor
    virtual ~SimuModuleFactory() {};

    // get modules: NULL when all modules are done
    SimuModule *GetModule() { return 0; };

private:
    // physically build the modules
    void MakeModule(const string& /* mName */) { ++fCurrent; };
    void MakeSequence(const string& /* mName */) {};

    // modules and sequences
    vector<SimuModule*> fModules;
    vector<Module*> fConcreteModules;

    // current module counter used in Get()
    size_t fCurrent;

   ClassDef(SimuModuleFactory,1);
};

bool abstractInVector() {
   // We are trying to test whether a vector of abstract pointer 
   // works.
   return false;
}

#ifdef __MAKECINT__
#pragma link C++ class SimuModule+;
#pragma link C++ class Module+;
#pragma link C++ class SimuModuleFactory+;
#endif
