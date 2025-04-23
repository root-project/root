
void execAutoLoadEntriesAsSelected(){

auto normName="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double>,ROOT::Math::DefaultCoordinateSystemTag>";
auto nameAsSelected="ROOT::Math::DisplacementVector2D<ROOT::Math::Cartesian2D<double> >";

auto cl1=TClass::GetClass(nameAsSelected);
auto cl2=TClass::GetClass(normName);

if (cl1 && cl1==cl2){
 std::cout << "The name as written in the linkDef," << nameAsSelected << ", triggered autoloading and the tclass entry is identical to the one specified by the normalised name.\n";
}

}
