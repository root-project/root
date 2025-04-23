#With gcc 3.1 on linux, the following crashed:
rootcint -f crashDict.C SkyMap.hh CrashLinkDef.hh 

#and then the following was happening:
g++ --shared -o libcxx.so crashDict.C -I$ROOTSYS/include && root.exe
root [] .L libcxx.so
Error: class,struct,union or type __gnu_cxx not defined
FILE:libcrash.so LINE:1
*** Interpreter error recovered ***

