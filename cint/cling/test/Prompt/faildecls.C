// RUN: cat %s | %cling

// "anonymous structs and classes must be class members"
typedef struct {int j;} T;
.q
