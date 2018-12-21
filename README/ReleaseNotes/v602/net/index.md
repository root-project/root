Introduce GZIP compression on the server side. Now one can request JSON (or any other data) in zipped form like:

    wget http://localhost:8080/Canvases/c1/root.json.gz

This solves problem with JSON using over network - such compressed file is about the same size as binary buffer.For that particular canvas (from hsimple.C example)

| format                   | size         |
| :----------------------- | -----------: |
| `root.json`              | 12994 bytes  |
| `root.json?compact=3`    |  8695 bytes  |
| `root.json.gz?compact=3` |  2071 bytes  |

It is factor 4 less data, transmitted between server and client.
"root.bin" request has also been modified. Now it is just data produced by TBufferFile without any additional headers.One can also compress such data with gzip:

    wget http://localhost:8080/Canvases/c1/root.bin.gz

