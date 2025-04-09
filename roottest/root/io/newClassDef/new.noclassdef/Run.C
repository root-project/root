{
gROOT->ProcessLine(".L namespace.so");
gROOT->ProcessLine(".L template.so ");
gROOT->ProcessLine(".L nstemplate.so ");
namespace_driver();
template_driver();
nstemplate_driver();

}
