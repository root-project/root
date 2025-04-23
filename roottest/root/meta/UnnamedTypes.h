#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-private-field"
#endif
typedef class {
   int i;
} UnnamedClass_t;

class {
   int j;
} UnnamedClassInstance;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
typedef struct {
   int k;
} UnnamedStruct_t;
