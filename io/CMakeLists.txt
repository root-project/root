add_subdirectory(io)  # special CMakeListst.txt 
add_subdirectory(sql)
add_subdirectory(xml)
if(xml)
  add_subdirectory(xmlparser)
endif()
if(castor)
  add_subdirectory(castor)
endif()
if(rfio)
  add_subdirectory(rfio)
endif()

