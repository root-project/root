{
int old = gInterpreter->SetClassAutoloading(kFALSE);
gInterpreter->LoadLibraryMap("lib/libEdm.rootmap");
gInterpreter->LoadLibraryMap("lib/libStrip.rootmap");
gInterpreter->LoadLibraryMap("lib/libCluster.rootmap");
gInterpreter->SetClassAutoloading(old);
}

