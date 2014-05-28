ROOTTEST_GENERATE_REFLEX_DICTIONARY(classes_rflx classes.h SELECTION classes_selection.xml)

ROOTTEST_ADD_TEST(root-meta-genreflex-ROOT-5626-execdummy MACRO execdummy.C OUTREF execdummy.ref)
