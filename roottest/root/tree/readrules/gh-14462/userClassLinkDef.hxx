#ifdef __CLING__

#pragma link C++ class userClass+;
// Note: we are testing here that the code property having spurious white space surrounding the open and closing
// brackets is properly handled (i.e. don't reformat it :) )
#pragma read sourceClass = "userClass" targetClass = "userClass" source = "" target = "transientMember" version = \
   "[1-]" code = "  {    transientMember = true;  }   "

#endif
