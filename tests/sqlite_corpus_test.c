/* Synthetic regression for the read-only SQLite lesson export. */
#define _POSIX_C_SOURCE 200809L
#include <sqlite3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>
static int sql(sqlite3 *d,const char *s){
 char *error=NULL;int r=sqlite3_exec(d,s,NULL,NULL,&error);
 if(error){fprintf(stderr,"%s\n",error);sqlite3_free(error);}
 return r==SQLITE_OK;
}
int main(void){
 char path[]="/tmp/stn-sqlite-XXXXXX",out[]="/tmp/stn-sqlite-out-XXXXXX";
 int fd=mkstemp(path),of=mkstemp(out);
 if(fd<0||of<0)return 1;
 close(fd);close(of);
 sqlite3 *db=NULL;int fail=0;
 if(sqlite3_open(path,&db)!=SQLITE_OK)return 1;
 fail|=!sql(db,"CREATE TABLE corpus(id INTEGER PRIMARY KEY,category TEXT,source TEXT,text TEXT,verification_status TEXT,source_record_id TEXT)");
 fail|=!sql(db,"INSERT INTO corpus(category,text,verification_status) VALUES"
 "('lesson:a.txt','Approved prose one.','operator-verified'),"
 "('lesson:b.txt','Approved prose one.','operator-verified'),"
 "('LESSON','Approved prose two.','operator-verified'),"
 "('LESSON','Unverified content.','unverified'),"
 "('observations','Forbidden content.','operator-verified'),"
 "('lesson:z.txt','Unapproved content.','verified')");
 sqlite3_close(db);
 if(fail){unlink(path);unlink(out);return 1;}
 pid_t pid=fork();if(pid<0)return 1;
 if(!pid){
  int dest=open(out,O_WRONLY|O_TRUNC);
  if(dest<0||dup2(dest,STDOUT_FILENO)<0)_exit(127);
  close(dest);execl("./sqlite_corpus","./sqlite_corpus",path,(char*)NULL);_exit(127);
 }
 int status=0;if(waitpid(pid,&status,0)<0||!WIFEXITED(status)||WEXITSTATUS(status))fail=1;
 FILE *f=fopen(out,"rb");if(!f)fail=1;
 else {
  char buf[1024];size_t n=fread(buf,1,sizeof(buf)-1,f);buf[n]=0;
  if(strcmp(buf,"Approved prose one.\n\nApproved prose two.\n\n"))fail=1;
  fclose(f);
 }
 unlink(path);unlink(out);
 printf("sqlite corpus isolation/deduplication: %s\n",fail?"FAIL":"PASS");
 return fail;
}
