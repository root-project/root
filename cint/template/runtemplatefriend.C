{
// fails due to CINT's autodict facility:
// when dict for shared_ptr is generated no dict for the templated constructor is requested
gROOT->ProcessLine(".autodict");
gROOT->ProcessLine(".x templatefriend.cxx");
gROOT->ProcessLine(".U templatefriend.cxx");
gROOT->ProcessLine(".x templatefriend.cxx+");
}
