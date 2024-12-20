# Latency tests for RWebWindow

Provide round-trip test under different conditions.
To run, execute `root "ping.cxx(10,0)"`, where first argument is number of connections tested and
second argument is running mode.

Can be tested:
0 - default communication, no extra threads
1 - minimal timer for THttpServer, should reduce round-trip significantly
2 - use special thread for process requests in THttpServer, web window also runs in the thread
3 - in addition to special THttpThread also window starts own thread
4 - let invoke webwindow callbacks in the civetweb threads, expert mode only

One also can perform same tests with longpoll emulation of web sockets, if adding 10 to second parameter

When running in batch mode, function blocked until 200 round-trip packets send by the client
or 50s elappsed. Therefore ping.cxx test can be used for RWebWindow functionality tests 
like `root -l -b "ping.cxx(10,2)" -q`
