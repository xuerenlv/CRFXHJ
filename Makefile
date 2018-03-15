CXX = g++

INCPATH = -I./src -I./deps/liblbfgs-1.10/include/
CFLAGS = -std=c++11 -msse2 -fPIC -O3 -Wall -finline-functions
LDFLAGS = -Wl,-lpthread


all: crf


CRF_OBJS = $(addprefix build/, main.o utils.o dataset.o feature.o model.o path.o node.o parallel.o)
LBFPS_A = $(addprefix deps/liblbfgs-1.10/lib/.libs/, liblbfgs.a)
crf: $(CRF_OBJS)
#	cd deps && tar -zxvf liblbfgs-1.10.tar.gz && cd liblbfgs-1.10 && ./configure && make && make install && cd ../../
	$(CXX) $(CFLAGS) $(INCPATH) -o $@ $(CRF_OBJS) $(LDFLAGS) $(LBFPS_A)



build/%.o: src/%.cc
	@mkdir -p $(@D)
	$(CXX) $(INCPATH) -std=c++0x -MM -MT build/$*.o $< > build/$*.d
	$(CXX) $(CFLAGS) $(INCPATH) -c $< -o $@


clean:
	rm -rf build crf
#	rm -rf ./deps/liblbfgs-1.10

