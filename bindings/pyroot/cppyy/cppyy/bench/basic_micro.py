import cppyy, gc, math, os, psutil, time

NLARGE = 20000000

process = psutil.Process(os.getpid())

def benchit(what, callf, N):
    print("running:", what)
    mpre = process.memory_info().rss/1024
    tpre = time.perf_counter()
    for i in range(N):
        callf()
    tpost = time.perf_counter()
    gc.collect()
    mpost = process.memory_info().rss/1024
    if tpost - tpre < 1.:
        print("  suggest increasing N by %dx" % math.ceil(1./(tpost-tpre)))
    print("  time:", (tpost - tpre)/N)
    if mpost == mpre:
        print("  memcheck passed")
    else:
        print("  memcheck FAILED:", mpre, mpost)

cppyy.cppdef("""
    void gfunc() {}

    class MyClass {
    public:
        void mfunc() {}
    };
""")


f = cppyy.gbl.gfunc
benchit("global function", f, NLARGE)

inst = cppyy.gbl.MyClass()
benchit("member function", inst.mfunc, NLARGE)
