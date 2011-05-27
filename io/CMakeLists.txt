add_subdirectory(io)  # special CMakeListst.txt 
add_subdirectory(sql)
add_subdirectory(xml)
if(xml)
  add_subdirectory(xmlparser)
endif()
