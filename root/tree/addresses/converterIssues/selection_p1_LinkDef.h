#ifdef __MAKECINT__
#pragma link C++ class EventFormat_p1+;
#pragma link C++ class edm::EventFormatElement+;
// #pragma link C++ class edm::EventFormat+;


//   <!-- Custom streamer for the event format class: -->
#pragma read sourceClass="EventFormat_p1" version="[1-]" \
         targetClass="edm::EventFormat" \
         source="std::vector<std::string> m_branchNames; std::vector<std::string> m_classNames; std::vector<uint32_t> m_branchHashes;" \
         target="" code = "{ newObj->clear(); \
         for( size_t i = 0; i < onfile.m_branchNames.size(); ++i ) { \
            newObj->add( edm::EventFormatElement( onfile.m_branchNames[ i ], \
                                                  onfile.m_classNames[ i ], \
                                                  onfile.m_branchHashes[ i ] ) ); \
         } }"
#endif
