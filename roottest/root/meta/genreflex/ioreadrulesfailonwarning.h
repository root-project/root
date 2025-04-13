#include <vector>
#include <Rtypes.h>

class UserData {
public:
   vector<double> posx;
   vector<double> posy;
   vector<double> posz;

   vector<double> posa;
   vector<double> posb;


   ClassDef(UserData,3);
};

#ifdef __ROOTCLING__
#pragma read sourceClass="UserData" targetClass="UserData" version="[1]" source="float posx" target="posx" code="{ posx.push_back( onfile.posx ); }"
#pragma read sourceClass="UserData" targetClass="UserData" version="[1]" source="float posy;" target="posy" code="{ posy.push_back( onfile.posy ); }"
#pragma read sourceClass="UserData" targetClass="UserData" version="[1]" source="float posz;" target="posz;" code="{ posz.push_back( onfile.posz ); }"

#pragma read sourceClass="UserData" targetClass="UserData" version="[1]" source="float posa; float posb;" target="posa; posb;" code="{posa.resize(1, onfile.posa); posb.resize(1, onfile.posb);}"

#endif

