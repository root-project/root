## \file
## \ingroup tutorial_sql
## \notebook -nodraw
## Fill run catalog with nfiles entries
##
## Based on sqlfill.C by Sergey Linev
##
## \macro_code
##
## \author Juan Fernando Jaramillo Botero

from ROOT import TSQLServer, TSQLResult, TStopwatch, gRandom


nfiles = 1000

ins = "INSERT INTO runcatalog VALUES ('%s', %d," \
      " %d, %d, %d, %10.2f, '%s', '%s', '1997-01-15 20:16:28'," \
      " '1999-01-15 20:16:28', '%s', '%s')"

tag = evt = 0

# open connection to MySQL server on localhost
db = TSQLServer.Connect("mysql://localhost/test", "nobody", "")

# first clean table of old entries
res = db.Query("DELETE FROM runcatalog")

# start timer
timer = TStopwatch()
timer.Start()

# fill run catalog
for i in range(nfiles):
    dataset = "testrun_%d" % i
    rawfile = "/v1/data/lead/test/run_%d.root" % i
    tag = int(gRandom.Rndm() * 10.)
    sql = ins % (dataset, i, evt, evt + 10000, tag, 25.5, "test", "lead",
                 rawfile, "test run dummy data")
    evt += 10000
    res = db.Query(sql)
    # print("%s" % sql)


# stop timer and print results
timer.Stop()
rtime = timer.RealTime()
ctime = timer.CpuTime()

print("")
print("%d files in run catalog" % nfiles)
print("RealTime=%f seconds, CpuTime=%f seconds" % (rtime, ctime))
