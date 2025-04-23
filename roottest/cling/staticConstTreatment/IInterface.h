
class InterfaceID {
 public:
    InterfaceID( unsigned long id,
                 unsigned long minor_id,
                 unsigned long major_id) : m_id( id ),
                                           m_major_ver( minor_id ),
                                           m_minor_ver( major_id ) { }
   unsigned long id() const { return m_id; }
   unsigned long majorVersion() const { return m_major_ver; }
   unsigned long minorVersion() const { return m_minor_ver; }
 private:
   unsigned long m_id;
   unsigned long m_major_ver;
   unsigned long m_minor_ver;
};