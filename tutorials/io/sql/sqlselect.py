## \file
## \ingroup tutorial_sql
##
## Query example to MySQL test database.
## Example of query by using the test database made in MySQL, you need the
## database test installed in localhost, with user nobody without password.
##
## Based on sqlselect.C by Sergey Linev
##
## \macro_code
##
## \author Juan Fernando Jaramillo Botero

from ROOT import TSQLServer, TSQLResult, TSQLRow, TStopwatch


db = TSQLServer.Connect("mysql://localhost/test", "nobody", "")

print("Server info: %s" % db.ServerInfo())

# list databases available on server
print("")
print("List all databases on server %s" % db.GetHost())
res = db.GetDataBases()
row = res.Next()
while row:
    print("%s" % row.GetField(0))
    row = res.Next()

# list tables in database "test" (the permission tables)
print('')
print('List all tables in database "test" on server %s' % db.GetHost())
res = db.GetTables("test")
row = res.Next()
while row:
    print("%s" % row.GetField(0))
    row = res.Next()

# list columns in table "runcatalog" in database "mysql"
print('')
print('List all columns in table "runcatalog" in database "test" on server %s' %
      db.GetHost())
res = db.GetColumns("test", "runcatalog")
row = res.Next()
while row:
    print("%s" % row.GetField(0))
    row = res.Next()

# start timer
timer = TStopwatch()
timer.Start()

# query database and print results
# sql = "select dataset,rawfilepath from test.runcatalog " \
#       "WHERE tag&(1<<2) AND (run=490001 OR run=300122)"
sql = "select count(*) from test.runcatalog " \
      "WHERE tag&(1<<2)"

res = db.Query(sql)

nrows = res.GetRowCount()
print("")
print("Got %d rows in result" % nrows)

nfields = res.GetFieldCount()
for i in range(nfields):
    print("%40s" % res.GetFieldName(i))
print("")
print("=" * (nfields * 40))
print("")

for i in range(nrows):
    row = res.Next()
    for j in range(nfields):
        print("%40s" % row.GetField(j))
    print("")

# stop timer and print results
timer.Stop()
rtime = timer.RealTime()
ctime = timer.CpuTime()

print("")
print("RealTime=%f seconds, CpuTime=%f seconds" % (rtime, ctime))
