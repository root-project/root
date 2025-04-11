
#include <map>
#include <string>
#include <list>
#include <iostream>
#include <typeinfo>

// #include "Algorithm/Algorithm.h"

using std::map;
using std::string;
using std::ostream;
using std::type_info;
using std::list;

template <class T> class genie__Wrapper {
public: 
   T val;
};

namespace genie {
   
   template <class T> class Wrapper {
   public: 
      T val;
   };
   
   class Algorithm;
   
   class RegistryItemI 
   {
   public:
      
      virtual const type_info & TypeInfo(void)  const        = 0;
      virtual void              Print(ostream& stream) const = 0;
      
   protected:
      
      RegistryItemI()          { }
      virtual ~RegistryItemI() { }
   };
   
   template<class T> class RegistryItem;
   template<class T>
   ostream & operator << (ostream & stream, const RegistryItem<T> & )
   { return stream; };
   
   template<class T> class RegistryItem;
   template<class T>
   ostream & operator >> (ostream & stream, const RegistryItem<T> & )
   { return stream; };
   
   template<class T> bool operator< (const RegistryItem<T> & , const RegistryItem<T> & ) 
   { return false; };
   
   template<class T> bool operator> (const RegistryItem<T> & , const RegistryItem<T> & ) 
   { return false; };
   
   template<class T> class RegistryItem : public RegistryItemI {
      
   public:
      
      RegistryItem() { };
      RegistryItem(T ) {};
      ~RegistryItem() { }
      
      const type_info & TypeInfo (void) const { return typeid(fItem); }
      const T &         Data     (void) const { return fItem;         }
      
      void Print(ostream& ) const {};
      
      friend ostream & operator<< <T>(ostream & stream, const RegistryItem<T> & rec); 
      friend ostream & operator>> <T>(ostream & stream, const RegistryItem<T> & rec); 
      friend bool operator< <T>(const RegistryItem<T> & lhs, const RegistryItem<T> & rhs); 
      friend bool operator> <T>(const RegistryItem<T> & lhs, const RegistryItem<T> & rhs); 
      
   private:
      
      T fItem;    
   };
   
   class Registry {
      
   public:
      
      Registry() {};
      Registry(const char * /* name */, bool /* isReadOnly */ = true) {};
      Registry(const Registry &) {};
      virtual ~Registry() {}
      
      //-- RegistryI interface implementation
      
      void   Lock        (void) {}
      void   UnLock      (void) {}
      bool   LockStatus  (void) const { return false; }
      void   Set         (const char * /* key */, bool         /* item */) {}
      void   Set         (string       /* key */, bool         /* item */) {}
      void   Set         (const char * /* key */, int          /* item */) {}
      void   Set         (string       /* key */, int          /* item */) {}
      void   Set         (const char * /* key */, double       /* item */) {}
      void   Set         (string       /* key */, double       /* item */) {}
      void   Set         (const char * /* key */, const char * /* item */) {}
      void   Set         (const char * /* key */, string       /* item */) {}
      void   Set         (string       /* key */, string       /* item */) {}
      void   Get         (const char * /* key */, bool &       /* item */) const {}
      void   Get         (string       /* key */, bool &       /* item */) const {}
      void   Get         (const char * /* key */, int &        /* item */) const {}
      void   Get         (string       /* key */, int &        /* item */) const {}
      void   Get         (const char * /* key */, double &     /* item */) const {}
      void   Get         (string       /* key */, double &     /* item */) const {}
      void   Get         (const char * /* key */, string &     /* item */) const {}
      void   Get         (string       /* key */, string &     /* item */) const {}
      int    NEntries    (void) const { return 0; }
      bool   Exists      (const char * /* key */) const {return false; }
      bool   Exists      (string       /* key */) const {return false; }
      bool   DeleteEntry (const char * /* key */) {return false; }
      bool   DeleteEntry (string       /* key */) {return false; }
      void   SetName     (const char * /* name */) {}
      void   SetName     (string /* name */) {}
      string Name        (void) const { return ""; }
      void   Print       (ostream & /* stream */) const {}
      
      friend ostream& operator<<(ostream& /* stream */, const Registry& registry);
      friend ostream& operator>>(ostream& /* stream */, const Registry& registry);
      
      void operator () (const char * /* key */,  int          /* item */) {}
      void operator () (string       /* key */,  int          /* item */) {}
      void operator () (const char * /* key */,  bool         /* item */) {}
      void operator () (string       /* key */,  bool         /* item */) {}
      void operator () (const char * /* key */,  double       /* item */) {}  
      void operator () (string       /* key */,  double       /* item */) {}
      void operator () (const char * /* key */,  string       /* item */) {}  
      void operator () (string       /* key */,  string       /* item */) {}
      void operator () (const char * /* key */,  const char * /* item */) {}
      
      Registry * operator =  (const Registry & ) { return this; }
      
   private:
      
      string fName;
      
      map<string, genie::RegistryItemI *> fRegistry;
   };
   
   ostream& operator<<(ostream& stream, const Registry& /* registry */) {
      return stream;
   }
   
   ostream& operator>>(ostream& stream, const Registry& /* registry */) {
      return stream;
   }
   
   class ConfigPool {
      
   public:
      
      static ConfigPool * Instance() { return 0; }
      
      Registry * FindRegistry(const Algorithm * ) const { return 0; }
      
      void Print(ostream &) const {}
      
      friend ostream& operator<<(ostream& stream, const ConfigPool & config_pool);
      friend ostream& operator>>(ostream& stream, const ConfigPool & config_pool);
      
   private:
      
      ConfigPool() {} 
      ConfigPool(const ConfigPool & ) {}
      virtual ~ConfigPool() {}
      
      bool LoadXMLConfig(void) { return false; }
      
      //   static ConfigPool * fInstance;
      
      map<string, genie::Registry *> fRegistryPool;    //-- algorithm/param_set -> Registry
   public:
      struct Cleaner {
         void DummyMethodAndSilentCompiler() { }
         ~Cleaner() {
         }
      };
	  public:
      friend struct Cleaner;
   };
   
   ostream& operator<<(ostream& stream, const ConfigPool & ) {
      return stream;
   }
   ostream& operator>>(ostream& stream, const ConfigPool & ) {
      return stream;
   }
   
}      // genie namespace



#ifdef __MAKECINT__

#pragma link C++ namespace genie;
#pragma link C++ class genie::ConfigPool;
#pragma link C++ class genie::Registry;
#pragma link C++ class genie::RegistryItem<bool>+;
#pragma link C++ class genie::RegistryItem<int>+;
#pragma link C++ class genie::RegistryItem<double>+;
#pragma link C++ class genie::RegistryItem<string>+;
// #pragma link C++ class genie::RegistryItem<list<double> >+; // wont work because the friend parsing does not handle multiple template param
// #pragma link C++ class genie::RegistryItem<genie::Wrapper<double> >+; // wont work because the friend parsing does not handle scope in  template param
#pragma link C++ class genie::RegistryItem<genie__Wrapper<double> >+; // wont work because the friend parsing does not handle scope in  template param
#pragma link C++ class genie::RegistryItemI;

#endif // __MAKECINT__
