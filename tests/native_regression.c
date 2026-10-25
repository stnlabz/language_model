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
#include <time.h>
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

static void service_test(const char *ckpt){
 int port=22000+(getpid()%20000);
 char portstr[16];snprintf(portstr,sizeof(portstr),"%d",port);
 pid_t pid=fork();
 if(pid<0){CHECK(0);return;}
 if(pid==0){execl("./llm_service","./llm_service","./micro_llm",ckpt,portstr,(char*)NULL);_exit(127);}
 int sock=-1;
 for(int attempt=0;attempt<20;attempt++){
  sock=socket(AF_INET,SOCK_STREAM,0);if(sock<0)break;
  struct sockaddr_in addr={0};addr.sin_family=AF_INET;addr.sin_port=htons((unsigned short)port);
  inet_pton(AF_INET,"127.0.0.1",&addr.sin_addr);
  if(connect(sock,(struct sockaddr*)&addr,sizeof(addr))==0)break;
  close(sock);sock=-1;struct timespec delay={0,100000000L};nanosleep(&delay,NULL);
 }
 if(sock>=0){
  const char *req="GET /health HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n";
  CHECK(send(sock,req,strlen(req),0)==(ssize_t)strlen(req));
  char reply[1024];ssize_t n=recv(sock,reply,sizeof(reply)-1,0);
  CHECK(n>0);
  if(n>0){reply[n]=0;CHECK(strstr(reply,"200 OK")!=NULL);}
  close(sock);
  sock=socket(AF_INET,SOCK_STREAM,0);
  if(sock>=0){
   struct sockaddr_in target={0};
   target.sin_family=AF_INET;
   target.sin_port=htons((unsigned short)port);
   inet_pton(AF_INET,"127.0.0.1",&target.sin_addr);
   if(connect(sock,(struct sockaddr*)&target,sizeof(target))==0){
    const char *payload="{\"prompt\":\"Hello\",\"max_bytes\":8,\"seed\":42}";
    char post[512];
    int size=snprintf(post,sizeof(post),"POST /generate HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n%s",strlen(payload),payload);
    CHECK(size>0&&(size_t)size<sizeof(post));
    if(size>0&&(size_t)size<sizeof(post)){
     CHECK(send(sock,post,(size_t)size,0)==size);
     char response_buf[2048]={0};size_t used=0;ssize_t received;
     while(used+1<sizeof(response_buf)&&(received=recv(sock,response_buf+used,sizeof(response_buf)-used-1,0))>0)used+=(size_t)received;
     response_buf[used]=0;
     CHECK(strstr(response_buf,"200 OK")!=NULL);
     CHECK(strstr(response_buf,"\"text\"")!=NULL);
    }
   }else CHECK(0);
   close(sock);
  }else CHECK(0);
 }else CHECK(0);
 kill(pid,SIGTERM);waitpid(pid,NULL,0);
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
 service_test(ckpt);
 unlink(src);unlink(bad);unlink(ckpt);rmdir(dir);
}
/* Optional controlled English memorization qualification.
 * Synthetic fixture only; never touches approved JSONL or real checkpoints.
 * A successful pipeline is not treated as a passed language probe. */
static int language_probe(void){
 char dir[]="/tmp/stn-language-probe-XXXXXX";
 if(!mkdtemp(dir)){perror("mkdtemp");return 1;}
 char corpus[512],ckpt[512];
 snprintf(corpus,sizeof(corpus),"%s/english.txt",dir);
 snprintf(ckpt,sizeof(ckpt),"%s/probe.stn",dir);
 FILE *f=fopen(corpus,"wb");
 if(!f){perror(corpus);rmdir(dir);return 1;}
 const char *pair="Question: What is an engineer? Answer: An engineer designs systems.\n\n";
 for(int n=0;n<100;n++)if(fputs(pair,f)==EOF){fclose(f);unlink(corpus);rmdir(dir);return 1;}
 if(fclose(f)){unlink(corpus);rmdir(dir);return 1;}
 char *train_argv[]={"./micro_llm","train",corpus,ckpt,"800",NULL};
 int status=run("./micro_llm",train_argv);
 if(status!=0){fprintf(stderr,"language_probe: training FAILED (%d)\n",status);unlink(corpus);unlink(ckpt);rmdir(dir);return 1;}
 int pipefd[2];if(pipe(pipefd)){perror("pipe");unlink(corpus);unlink(ckpt);rmdir(dir);return 1;}
 pid_t pid=fork();
 if(pid==0){
  close(pipefd[0]);dup2(pipefd[1],STDOUT_FILENO);close(pipefd[1]);
  execl("./micro_llm","./micro_llm","generate",ckpt,"Question: What is an engineer? Answer:","32","913",(char*)NULL);
  _exit(127);
 }
 close(pipefd[1]);char output[256]={0};size_t used=0;
 while(used<sizeof(output)-1){
  ssize_t n=read(pipefd[0],output+used,sizeof(output)-1-used);
  if(n<=0)break;used+=(size_t)n;
 }
 close(pipefd[0]);int child=0;
 if(pid<0||waitpid(pid,&child,0)<0||!WIFEXITED(child)||WEXITSTATUS(child)!=0)status=1;
 else{
  output[used]=0;
  printf("language_probe generated: %s",output);
  if(strncmp(output," An engineer designs systems.",29)!=0){
   fputs("language_probe: FAIL - English continuation not reproduced\n",stderr);
   status=1;
  }else{puts("language_probe: PASS - memorized controlled English continuation");status=0;}
 }
 unlink(corpus);unlink(ckpt);rmdir(dir);
 return status!=0;
}
int main(int argc,char **argv){
 if(argc==2&&!strcmp(argv[1],"--language-probe"))return language_probe();
 if(argc!=1){fputs("usage: native_regression [--language-probe]\\n",stderr);return 2;}
 json_tests();end_to_end();
 if(failures){fprintf(stderr,"%d failures\\n",failures);return 1;}
 puts("native regression tests passed");return 0;
}
