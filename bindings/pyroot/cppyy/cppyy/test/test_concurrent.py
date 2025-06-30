import pytest
from pytest import raises, skip, mark
from support import IS_MAC_ARM


class TestCONCURRENT:

    def setup_class(cls):
        import cppyy

        cls.data = [3.1415, 2.7183, 1.4142, 1.3807, -9.2848]

        cppyy.cppdef("""\
        // as recommended by:
        // https://docs.python.org/3/c-api/intro.html#include-files
        #define PY_SSIZE_T_CLEAN
        #include "Python.h"

        namespace Workers {
            double calc(double d) { return d*42.; }
        }""")

        cppyy.gbl.Workers.calc.__release_gil__ = True

    @mark.skip
    def test01_simple_threads(self):
        """Run basic Python threads"""

        import cppyy
        import threading

        cppyy.gbl.Workers.calc.__release_gil__ = True

        threads = []
        for d in self.data:
            threads.append(threading.Thread(target=cppyy.gbl.Workers.calc, args=(d,)))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    @mark.skip
    def test02_futures(self):
        """Run with Python futures"""

        import cppyy

        try:
            import concurrent.futures
        except ImportError:
            skip("module concurrent is not installed")

        total = 0.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.data)) as executor:
            futures = [executor.submit(cppyy.gbl.Workers.calc, d) for d in self.data]
            for f in concurrent.futures.as_completed(futures):
                total += f.result()
        assert round(total+26.4642, 8) == 0.0

    def test03_timeout(self):
        """Time-out with threads"""

        return
        import cppyy
        import threading, time

        cppyy.cppdef("""\
        namespace test12_timeout {
            bool _islive = false; volatile bool* islive = &_islive;
            bool _stopit = false; volatile bool* stopit = &_stopit;
        }""")

        cppyy.gbl.gInterpreter.ProcessLine.__release_gil__ = True
        cmd = r"""\
           *test12_timeout::islive = true;
           while (!*test12_timeout::stopit);
        """

        t = threading.Thread(target=cppyy.gbl.gInterpreter.ProcessLine, args=(cmd,))
        t.start()

      # have to give ProcessLine() time to actually start doing work
        while not cppyy.gbl.test12_timeout.islive:
            time.sleep(0.1)     # in seconds

      # join the thread with a timeout after 0.1s
        t.join(0.1)             # id.

        if t.is_alive():        # was timed-out
            cppyy.gbl.test12_timeout.stopit[0] = True

    def test04_cpp_threading_with_exceptions(self):
        """Threads and Python exceptions"""

        if IS_MAC_ARM:
            skip("JIT exceptions can not be caught in JITed code on Mac ARM")

        import cppyy

        cppyy.include("CPyCppyy/PyException.h")

        cppyy.cppdef("""namespace thread_test {
        #include <thread>

        struct consumer {
            virtual ~consumer() {}
            virtual void process(int) = 0;
        };

        struct worker {
            worker(consumer* c) : cons(c) {
                // Get the main interpreter state state to spawn new thread states
                PyThreadState* state = PyThreadState_Get();
                interpreterState = state->interp;
            }
            ~worker() { wait(); }

            void start() {
                t = std::thread([this] {
                    int counter = 0;

                    // Each thread needs a Python state object
                    // Instead of using the higher-level PyGILState_Ensure and
                    // PyGILState_Release functions, use the PyThreadState API
                    // directly so that we only need to create one single
                    // PyThreadState that can be restored and released in the
                    // "hot loop".
                    PyThreadState *pystate = PyThreadState_New(this->interpreterState);

                    while (counter++ < 10)
                        try {
                            PyEval_RestoreThread(pystate);
                            cons->process(counter);
                            PyEval_SaveThread();
                        } catch (CPyCppyy::PyException& e) {
                            err_msg = e.what();
                            PyEval_SaveThread();
                            return;
                        }

                    PyEval_RestoreThread(pystate);
                    PyThreadState_Clear(pystate);
                    PyEval_SaveThread();

                    PyThreadState_Delete(pystate);
                });
            }

            void wait() {
                if (t.joinable())
                    t.join();
            }

            PyInterpreterState* interpreterState = nullptr;
            std::thread t;
            consumer* cons = nullptr;
            std::string err_msg;
        }; }""")

        ns = cppyy.gbl.thread_test
        consumer = ns.consumer
        worker = ns.worker
        worker.wait.__release_gil__ = True

        class C(consumer):
            count = 0
            def process(self, c):
                self.count += 1

        c = C()
        assert c.count == 0

        w = worker(c)
        w.start()
        w.wait()

        assert c.count == 10

        class C(consumer):
            count = 0
            def process(self, c):
                raise RuntimeError("all wrong")

        c = C()

        w = worker(c)
        w.start()
        w.wait()

        assert "RuntimeError" in w.err_msg
        assert "all wrong"    in w.err_msg

    def test05_float2d_callback(self):
        """Passing of 2-dim float arguments"""

        import cppyy

        cppyy.cppdef("""\
        namespace FloatDim2 {
        #include <thread>

        struct Buffer {
            Buffer() = default;

            void setData(float** newData) {
                data = newData;
            }

            void setSample(int newChannel, int newSample, float value) {
                data[newChannel][newSample] = value;
            }

            float** data = nullptr;
        };

        struct Processor {
            virtual ~Processor() = default;
            virtual void process(float** data, int channels, int samples) = 0;
        };

        void callback(Processor& p) {
            std::thread t([&p] {
                int channels = 2;
                int samples = 32;

                float** data = new float*[channels];
                for (int i = 0; i < channels; ++i)
                    data[i] = new float[samples];

                // Set Python thread because we call back into Python
                PyGILState_STATE gstate;
                gstate = PyGILState_Ensure();
                p.process(data, channels, samples);
                PyGILState_Release(gstate);

                for (int i = 0; i < channels; ++i)
                    delete[] data[i];

                delete[] data;
            });

            t.join();
        } }""")

        cppyy.gbl.FloatDim2.callback.__release_gil__ = True

        class Processor(cppyy.gbl.FloatDim2.Processor):
            buffer = cppyy.gbl.FloatDim2.Buffer()

            def process(self, data, channels, samples):
                self.buffer.setData(data)

                try:
                    for c in range(channels):
                        for s in range(samples):
                            self.buffer.setSample(c, s, 0.0) # < used to crash here
                except Exception as e:
                    print(e)

        p = Processor()
        cppyy.gbl.FloatDim2.callback(p)

    def test06_overload_reuse_in_threads(self):
        """Threads reuse overload objects; check for clashes"""

        import cppyy
        import threading

        cppyy.cppdef("""\
        namespace CPPOverloadReuse {
        class Simulation1 {
        public:
            virtual void set_something(std::map<std::string, std::string>, std::string) {}
        }; }""")

        def test():
            o = {"a": "b"}        # causes a temporary to be created
            simulation = cppyy.gbl.CPPOverloadReuse.Simulation1()
            simulation.set_something(o, ".")

        threads = [threading.Thread(target=test) for i in range(0, 100)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

    @mark.skip()
    def test07_overload_reuse_in_threads_wo_gil(self):
        """Threads reuse overload objects; check for clashes if no GIL"""

        if IS_MAC_ARM:
          # the culprit here is occasional std::system_error if a thread can not be joined
            skip("JIT exceptions can not be caught in JITed code on Mac ARM")

        import cppyy
        import threading

        cppyy.cppdef("""\
        namespace CPPOverloadReuse {
        class Simulation2 {
            int fCount;
        public:
            Simulation2(int count) : fCount(count) {}
            virtual int do_something() { return fCount; }
        }; }""")

        cppyy.gbl.CPPOverloadReuse.Simulation2.do_something.__release_gil__ = True

        class State(object):
            lock = threading.Lock()
            c1, c2, c3 = 0, 0, 0

        def test(i, state=State):
            #global c1, c2, c3, lock

            simulation = cppyy.gbl.CPPOverloadReuse.Simulation2(i)
            res = simulation.do_something()

            with state.lock:
                state.c3 += state.c1
                state.c2 += res
                state.c1 += 1

        threads = [threading.Thread(target=test, args=[i]) for i in range(0, 1000)]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        assert State.c1 == 1000
        assert State.c2 == State.c3


if __name__ == "__main__":
    exit(pytest.main(args=['-sv', '-ra', __file__]))
