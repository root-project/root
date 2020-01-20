#define SCHEMA_CHANGE
#ifdef SCHEMA_CHANGE
/*


rootcint -f Simple_dict.cpp -p -D"SCHEMA_CHANGE" Simple.h Simple_LinkDef.h
cl.exe /c /DLL /MD /Z7 /GR /GX /D"SCHEMA_CHANGE" Simple_dict.cpp /I"%ROOTSYS%/include"
link.exe Simple_dict.obj /libpath:"%ROOTSYS%\lib" libCint.lib libCore.lib /DLL -out:SimpleChange.dll


*/
namespace AddSpace {
#endif
class Simple {
  public:
    float   m_data0;
    Simple() : m_data0(0.)  {  }
    virtual ~Simple() {}
    void set(int i, int )  {
      m_data0 = float(i+1);
    }
    void dump() const  {
    }
};
#ifdef SCHEMA_CHANGE
}
#endif
