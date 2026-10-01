#include <gtest/gtest.h>
#include "TSQLiteServer.h"
#include "TSQLiteResult.h"
#include "TSQLiteRow.h"
#include "TSQLiteStatement.h"
#include "TSQLColumnInfo.h"
#include "TSQLTableInfo.h"
#include "TSystem.h"
#include "TList.h"

#include <memory>
#include <string>

#include "ROOT/TestSupport.hxx"

TEST(SQLiteTest, ConnectAndQuery)
{
   ROOT::TestSupport::FileRaii dbFile{"test_sqlite.db"};

   std::string uri = std::string("sqlite://") + dbFile.GetPath();
   std::unique_ptr<TSQLServer> server(TSQLServer::Connect(uri.c_str(), "", ""));
   ASSERT_NE(server, nullptr);
   ASSERT_FALSE(server->IsZombie());

   // Test Exec (CREATE and INSERT)
   EXPECT_TRUE(server->Exec("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);"));
   EXPECT_TRUE(server->Exec("INSERT INTO test_table (id, name) VALUES (1, 'Alice');"));
   EXPECT_TRUE(server->Exec("INSERT INTO test_table (id, name) VALUES (2, 'Bob');"));

   // Test Query
   std::unique_ptr<TSQLResult> res(server->Query("SELECT id, name FROM test_table ORDER BY id;"));
   ASSERT_NE(res, nullptr);

   EXPECT_EQ(res->GetFieldCount(), 2);
   EXPECT_STREQ(res->GetFieldName(0), "id");
   EXPECT_STREQ(res->GetFieldName(1), "name");

   // Row 1
   std::unique_ptr<TSQLRow> row1(res->Next());
   ASSERT_NE(row1, nullptr);
   EXPECT_STREQ(row1->GetField(0), "1");
   EXPECT_STREQ(row1->GetField(1), "Alice");

   // Row 2
   std::unique_ptr<TSQLRow> row2(res->Next());
   ASSERT_NE(row2, nullptr);
   EXPECT_STREQ(row2->GetField(0), "2");
   EXPECT_STREQ(row2->GetField(1), "Bob");

   // No more rows
   EXPECT_EQ(res->Next(), nullptr);
}

TEST(SQLiteTest, TableInfo)
{
   ROOT::TestSupport::FileRaii dbFile{"test_sqlite2.db"};

   std::string uri = std::string("sqlite://") + dbFile.GetPath();
   std::unique_ptr<TSQLServer> server(TSQLServer::Connect(uri.c_str(), "", ""));
   ASSERT_NE(server, nullptr);
   EXPECT_TRUE(server->Exec("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT);"));

   std::unique_ptr<TSQLResult> tables(server->GetTables("main"));
   ASSERT_NE(tables, nullptr);
   std::unique_ptr<TSQLRow> trow(tables->Next());
   ASSERT_NE(trow, nullptr);
   EXPECT_STREQ(trow->GetField(0), "test_table");

   std::unique_ptr<TSQLTableInfo> info(server->GetTableInfo("test_table"));
   ASSERT_NE(info, nullptr);
   auto columns = info->GetColumns();
   ASSERT_NE(columns, nullptr);
   EXPECT_EQ(columns->GetSize(), 2);
}

TEST(SQLiteTest, PreparedStatements)
{
   ROOT::TestSupport::FileRaii dbFile{"test_sqlite3.db"};

   std::string uri = std::string("sqlite://") + dbFile.GetPath();
   std::unique_ptr<TSQLServer> server(TSQLServer::Connect(uri.c_str(), "", ""));
   ASSERT_NE(server, nullptr);
   EXPECT_TRUE(server->Exec("CREATE TABLE test_table (id INTEGER PRIMARY KEY, value REAL);"));

   std::unique_ptr<TSQLStatement> stmt(server->Statement("INSERT INTO test_table (id, value) VALUES (?, ?);"));
   ASSERT_NE(stmt, nullptr);

   for (int i = 1; i <= 3; ++i) {
      EXPECT_TRUE(stmt->NextIteration());
      EXPECT_TRUE(stmt->SetInt(0, i));
      EXPECT_TRUE(stmt->SetDouble(1, i * 1.5));
   }
   EXPECT_TRUE(stmt->Process());

   std::unique_ptr<TSQLResult> res(server->Query("SELECT COUNT(*) FROM test_table;"));
   ASSERT_NE(res, nullptr);
   std::unique_ptr<TSQLRow> countRow(res->Next());
   ASSERT_NE(countRow, nullptr);
   EXPECT_STREQ(countRow->GetField(0), "3");
}
