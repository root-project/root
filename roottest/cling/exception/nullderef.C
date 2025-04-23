// ROOT-8832

int nullderef() {
  int *p = (int*)0x1;
  return *p;
}
