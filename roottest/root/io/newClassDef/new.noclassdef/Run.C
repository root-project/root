{
    gSystem->Load("libIoNewClassNewNoDefnamespace");
    gSystem->Load("libIoNewClassNewNoDeftemplate");
    gSystem->Load("libIoNewClassNewNoDefnstemplate");

    namespace_driver();
    template_driver();
    nstemplate_driver();
}
