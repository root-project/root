ROOT_ADD_ROOTTEST(cint-array-1 MACRO  runarray1test.C
                               OUTREF array1test.ref
                               OUTCNV /usr/bin/sed -e "s/0x[^ ]*//g" -e "s/[A-Za-z]:.*array1.C/array1.C/g")
                                  

