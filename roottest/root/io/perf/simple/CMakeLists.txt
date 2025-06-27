
# exclude vectorsptrtclass - it uses already private method

foreach(name clonestclass
listMWclass
listMWtclass
listOWclass
listOWtclass
oneclass
onetclass
readleaves
setMWclass
vectorint
vectorintInside
vectorMWclass
vectorMWtclass
vectorMWthit
vectorOWclass
vectorOWtclass
vectorptrtclass
vectortclass)
  ROOTTEST_ADD_TEST(${name}
                  MACRO ${name}.C+)
endforeach()
