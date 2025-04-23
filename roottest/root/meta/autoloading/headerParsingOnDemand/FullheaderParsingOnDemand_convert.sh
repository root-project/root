grep -v "Warning in <TInterpreter::RegisterModule()>: Header" \
| grep -v "status 0" | grep -v "memory" \
| grep -v "__functional_base"|grep -v "string" |grep -v TRint \
|grep -v "dtor called for" \
|grep -v ">>> RSS" | grep -v ">>> VSIZE" \
|grep -v "StreamerInfo"

