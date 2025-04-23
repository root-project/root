#include <vector>
#include "Rtypes.h"
class MemberMyObject {
public:
	MemberMyObject() {}
	MemberMyObject(int allocated, int filled) : v(allocated) {
		for (int i = 0; i < filled; i++) {
			v[i] = i;
		} 
                fprintf(stderr,"vector allocated %d\n",allocated);
                fprintf(stderr,"vector filled %d\n",filled);
                fprintf(stderr,"vector length %d\n",v.size());
	}
	std::vector<int> v;
//	ClassDef(MemberMyObject,1)
};
