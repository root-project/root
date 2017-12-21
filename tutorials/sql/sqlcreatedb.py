## \file
## \ingroup tutorial_sql
## \notebook -nodraw
## Create a runcatalog table in a MySQL test database.
##
## Based on the code sqlcreatedb.C by Sergey Linev
##
## \macro_code
##
## \author Juan Fernando Jaramillo Botero

from ROOT import TSQLServer


# read in runcatalog table definition
fp = open("runcatalog.sql", "r")
sql = fp.read()
fp.close()

# open connection to MySQL server on localhost
db = TSQLServer.Connect("mysql://localhost/test", "nobody", "")

# create new table (delete old one first if exists)
res = db.Query("DROP TABLE runcatalog")

res = db.Query(sql)
