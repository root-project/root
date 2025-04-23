#include <vector>

class MuonTileID 
  {
  public:

    MuonTileID() : m_id(0){}
    virtual ~MuonTileID() {}
  
    void setID(unsigned int id) { m_id = id; }
  
  #ifndef _WIN32
    /// operator new
    static void* operator new ( size_t )
    {
      static size_t i = 0;
      static std::vector<unsigned int> array(100);
      return &array[i++];
    }
  
    /// placement operator new
    /// it is needed by libstdc++ 3.2.3 (e.g. in std::vector)
    /// it is not needed in libstdc++ >= 3.4
    static void* operator new ( size_t size, void* pObj )
    {
      return ::operator new (size,pObj);
    }
  
    /// operator delete
    static void operator delete ( void* )
    {
    }
  
    /// placement operator delete
    /// not sure if really needed, but it does not harm
    static void operator delete ( void* p, void* pObj )
    {
      ::operator delete (p, pObj);
    }
  #endif

  private:
    unsigned int m_id;
  };


MuonTileID getID() {
  return MuonTileID();
}

