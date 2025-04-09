#ifdef __CINT__

#pragma link C++ class Belle2::RelationsInterface<TObject>+;
#pragma link C++ class Belle2::ExtHit+;
#pragma link C++ class Belle2+;

#pragma read \
sourceClass="Belle2::ExtHit" source="TMatrixDSym m_Covariance" version="[-3]" \
targetClass="Belle2::ExtHit" target="m_Cov" \
code = "{ \
int k = 0; \
if (0) onfile.m_Covariance.Dump(); \
for (int i = 0; i < 6; ++i) { \
for (int j = 0; j <= i; ++j) { \
if (1) m_Cov[k++] = onfile.m_Covariance(i,j); \
} \
} \
}"

#endif
