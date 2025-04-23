#define a __LINE__
#define b(a1) c ## a1
#define unique prefix #__LINE__

#define LINE 13

#define join2(one,two) one##two
#define join1(one,two) join2(one,two)
#define join (one,two) join1(one,two)

#define unique(x) join1(x, __LINE__ )

#define unique2(x) join1(x,LINE)

int unique(prefix);
int unique2(prefix2);
