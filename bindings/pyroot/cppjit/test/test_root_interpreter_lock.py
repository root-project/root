class TestROOTINTERPRETERLOCK:
    def test01_declare_holds_the_interpreter_lock(self):
        """cppdef runs under gInterpreterMutex once ROOT thread safety is on"""

        import cppjit

        cppjit.gbl.ROOT.EnableThreadSafety()
        cppjit.cppdef(r"""
            #include "TInterpreter.h"
            #include "TVirtualMutex.h"
            #include <atomic>
            #include <chrono>
            #include <thread>

            namespace lock_probe {
                std::atomic<bool> in_decl{false}, done{false}, saw_in_decl{false};

                // Waits until a declaration is being processed, then takes the
                // interpreter lock: when the declaring thread holds it, this
                // only succeeds once the declaration has finished.
                void contend() {
                    while (!in_decl)
                        std::this_thread::yield();
                    R__LOCKGUARD(gInterpreterMutex);
                    saw_in_decl = in_decl.load();
                    done = true;
                }

                // Static initializer of the probe declaration: gives the
                // contender up to 300 ms to acquire the lock meanwhile.
                int spin() {
                    in_decl = true;
                    auto t0 = std::chrono::steady_clock::now();
                    while (!done && std::chrono::steady_clock::now() - t0 <
                                        std::chrono::milliseconds(300))
                        std::this_thread::yield();
                    in_decl = false;
                    return 0;
                }

                std::thread* start() { return new std::thread(contend); }
                bool contender_saw_declaration() { return saw_in_decl; }
            }""")

        thread = cppjit.gbl.lock_probe.start()
        cppjit.cppdef("namespace lock_probe { int marker = spin(); }")
        thread.join()

        assert not cppjit.gbl.lock_probe.contender_saw_declaration()
