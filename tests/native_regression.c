/* Native C regression: JSON decoding, approved-data learning, checkpoint
 * and loopback service health/generation. Runs entirely on temporary fixtures.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../json_fields.h"
static int failures=0;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x);failures++;}}while(0)
static void json_tests(void){
 char answer[256]={0},approved[8]={0},number[20]={0};
 JField f[]={{answer,sizeof(answer),"response",0,0},{approved,sizeof(approved),"approved",0,0},{number,sizeof(number),"max_bytes",0,0}};
 CHECK(jfields("{\"response\":\"Hello\\nworld\",\"approved\":true,\"max_bytes\":64}",f,3)==0);
 CHECK(!strcmp(answer,"Hello\nworld"));
 CHECK(f[1].type==2&&f[2].type==5&&!strcmp(number,"64"));
 JField g[]={{answer,sizeof(answer),"response",0,0}};
 CHECK(jfields("{\"response\":\"\\uD83D\\uDE00\"}",g,1)==0);
 CHECK(strlen(answer)==4);
 CHECK(jfields("{\"response\":\"\\uD83D\"}",g,1)!=0);
 CHECK(jfields("{\"response\":\"\\q\"}",g,1)!=0);
 CHECK(jfields("{\"response\":\"x\",}",g,1)!=0);
 CHECK(jfields("{\"response\":\"x\",\"response\":\"y\"}",g,1)!=0);
 CHECK(jfields("{\"response\":{\"nested\":true}}",g,1)!=0);
}
static int run(const char *exe,char *const argv[]){
 pid_t pid=fork();if(pid<0)return -1;
 if(pid==0){execv(exe,argv);_exit(127);}
 int status=0;if(waitpid(pid,&status,0)<0)return -1;
 return WIFEXITED(status)?WEXITSTATUS(status):-1;
}
static void end_to_end(void){
 char dir[]="/tmp/stn-native-test-XXXXXX";
 CHECK(mkdtemp(dir)!=NULL);
 char src[512],ckpt[512],bad[512];snprintf(src,sizeof(src),"%s/data.jsonl",dir);
 snprintf(ckpt,sizeof(ckpt),"%s/model.stn",dir);
 snprintf(bad,sizeof(bad),"%s/bad.jsonl",dir);
 FILE *f=fopen(src,"wb");CHECK(f!=NULL);if(!f)return;
 for(int i=0;i<60;i++)fprintf(f,"{\"input\":\"Test %d\",\"response\":\"Native C language training example number %d.\",\"approved\":true}\n",i,i);
 fprintf(f,"{\"input\":\"REJECT\",\"response\":\"This is not approved.\",\"approved\":false}\n");
 fclose(f);
 char *argv[]={"./gru_learn",src,ckpt,"1","./micro_llm",NULL};
 CHECK(run("./gru_learn",argv)==0);
 CHECK(access(ckpt,R_OK)==0);
 CHECK(run("./gru_learn",argv)!=0);/* No checkpoint overwrite. */
 f=fopen(bad,"wb");CHECK(f!=NULL);if(f){
  fputs("{\"input\":\"broken\",\"approved\":true\n",f);fclose(f);
  char missing[512];snprintf(missing,sizeof(missing),"%s/no-model.stn",dir);
  char *badargv[]={"./gru_learn",bad,missing,"1","./micro_llm",NULL};
  CHECK(run("./gru_learn",badargv)!=0);
  CHECK(access(missing,F_OK)!=0);
 }
 char *gen[]={"./micro_llm","generate",ckpt,"Hello","8","42",NULL};
 CHECK(run("./micro_llm",gen)==0);
 unlink(src);unlink(bad);unlink(ckpt);rmdir(dir);
}
int main(void){json_tests();end_to_end();if(failures){fprintf(stderr,"%d failures\n",failures);return 1;}puts("native regression tests passed");return 0;}
