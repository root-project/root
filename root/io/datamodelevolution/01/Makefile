

#===[ general rules ]=========================================================#

.PHONY: all
all:

#=============================================================================#

#===[ test1.x ]===============================================================#

test1_OBJS  = test1.o
test1_LIBS  = -L. -lRIO -lCore -lCint -lTree -lNet -lMathCore -lThread -pthread
OBJS       += $(test1_OBJS)
PROGS      += test1.x
test1.x     : $(test1_OBJS) | $(DICTS)

#=============================================================================#

#===[ test2.x ]===============================================================#

test2_OBJS  = test2.o
test2_LIBS  = -L. -lRIO -lCore -lCint -lTree -lNet -lMathCore -lThread -pthread
OBJS       += $(test2_OBJS)
PROGS      += test2.x
test2.x     : $(test2_OBJS)

#=============================================================================#

#===[ test3.x ]===============================================================#

test3_OBJS  = test3.o
test3_LIBS  = -L. -lRIO -lCore -lCint -lTree -lNet -lMathCore -lThread -pthread
OBJS       += $(test3_OBJS)
PROGS      += test3.x
test3.x     : $(test3_OBJS) 

#=============================================================================#


#===[ dictionaries ]==========================================================#

DICTS  += libDataModelV1_dictREFLEX.so libDataModelV1_dictCINT.so
DICTS  += libDataModelV2_dictREFLEX.so libDataModelV2_dictCINT.so

#=============================================================================#

include ../common/rules.mk
