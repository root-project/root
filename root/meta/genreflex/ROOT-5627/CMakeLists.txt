ROOT_REFLEX_GENERATE_DICTIONARY(classes_dictrflx classes.h SELECTION classes_selection.xml)

ROOT_ADD_ROOTTEST(root-meta-genreflex-ROOT-5627-exec5627 MACRO exec5627.C OUTREF exec5627.ref)
