/* STN-LABZ read-only SQLite corpus adapter.
 * Deliberately exports only operator-verified lesson prose.
 * No access to context, observations, messages, or identity records.
 * It does not modify Digit, its database, or trigger training.
 */
#define _POSIX_C_SOURCE 200809L
#include <sqlite3.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_BYTES 50000000u
static int export_corpus(const char *path){
 sqlite3 *db=NULL; sqlite3_stmt *stmt=NULL;
 int flags=SQLITE_OPEN_READONLY|SQLITE_OPEN_FULLMUTEX;
 if(sqlite3_open_v2(path,&db,flags,NULL)!=SQLITE_OK){
  fprintf(stderr,"SQLite open failed: %s\n",db?sqlite3_errmsg(db):"out of memory");
  if(db)sqlite3_close(db);
  return 1;
 }
 /* Refuse databases without the expected schema; no writes or migrations. */
 const char *sql=
  "SELECT MIN(id), text FROM corpus "
  "WHERE verification_status='operator-verified' "
  "AND (category='LESSON' OR category GLOB 'lesson:*') "
  "AND text IS NOT NULL AND length(trim(text))>0 "
  "GROUP BY text ORDER BY MIN(id)";
 if(sqlite3_prepare_v2(db,sql,-1,&stmt,NULL)!=SQLITE_OK){
  fprintf(stderr,"Corpus query failed: %s\n",sqlite3_errmsg(db));
  sqlite3_close(db);return 1;
 }
 size_t total=0,records=0;
 int rc;
 while((rc=sqlite3_step(stmt))==SQLITE_ROW){
  const unsigned char *value=sqlite3_column_text(stmt,1);
  int bytes=sqlite3_column_bytes(stmt,1);
  if(!value||bytes<1)continue;
  if(memchr(value,0,(size_t)bytes)){
   fputs("Corpus contains NUL byte; refusing malformed export\n",stderr);
   rc=SQLITE_MISMATCH;break;
  }
  /* Normalize boundary only, leaving lesson text unchanged. */
  size_t n=(size_t)bytes;
  while(n && (value[n-1]=='\n'||value[n-1]=='\r'))n--;
  if(n<2)continue;
  if(n>MAX_BYTES-total-2){
   fputs("Approved corpus exceeds 50MB export limit\n",stderr);
   rc=SQLITE_TOOBIG;break;
  }
  if(fwrite(value,1,n,stdout)!=n||fwrite("\n\n",1,2,stdout)!=2){
   fputs("Corpus export write failure\n",stderr);
   rc=SQLITE_IOERR;break;
  }
  total+=n+2;records++;
 }
 int ok=(rc==SQLITE_DONE)&&records>0&&fflush(stdout)==0;
 if(rc==SQLITE_DONE&&records==0)fputs("No operator-verified lesson records\n",stderr);
 if(rc!=SQLITE_DONE&&rc!=SQLITE_TOOBIG&&rc!=SQLITE_MISMATCH&&rc!=SQLITE_IOERR)
  fprintf(stderr,"Corpus iteration failed: %s\n",sqlite3_errmsg(db));
 if(sqlite3_finalize(stmt)!=SQLITE_OK)ok=0;
 if(sqlite3_close(db)!=SQLITE_OK)ok=0;
 if(ok)fprintf(stderr,"sqlite_corpus records=%zu bytes=%zu duplicates_skipped_by_text=enabled\n",records,total);
 return ok?0:1;
}
int main(int argc,char **argv){
 if(argc!=2){fputs("usage: sqlite_corpus READ_ONLY_SQLITE_DB > corpus.txt\n",stderr);return 2;}
 return export_corpus(argv[1]);
}
