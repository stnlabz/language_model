/* llm_service: native C loopback HTTP adapter to versioned model checkpoint.
 * Explicit launch only. No systemd installation, no agent integrations.
 */
#define _POSIX_C_SOURCE 200809L
#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include "json_fields.h"
#define MAX_REQ 16384
static int sendall(int fd,const char *buf,size_t size){
 while(size){ssize_t n=send(fd,buf,size,MSG_NOSIGNAL);if(n<=0)return -1;buf+=n;size-=(size_t)n;}return 0;
}
static void response(int fd,int code,const char *body){
 const char *reason=code==200?"OK":code==400?"Bad Request":code==404?"Not Found":code==413?"Payload Too Large":"Service Unavailable";
 char header[256];
 int n=snprintf(header,sizeof(header),"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",code,reason,strlen(body));
 if(n>0&&(size_t)n<sizeof(header)){sendall(fd,header,(size_t)n);sendall(fd,body,strlen(body));}
}
static void escaped(char *dest,size_t cap,const char *src,size_t n){
 size_t pos=0;dest[pos++]='"';
 for(size_t i=0;i<n&&pos+8<cap;i++){
  unsigned char c=(unsigned char)src[i];
  if(c=='"'||c=='\\'){dest[pos++]='\\';dest[pos++]=(char)c;}
  else if(c=='\n'){dest[pos++]='\\';dest[pos++]='n';}
  else if(c=='\r'){dest[pos++]='\\';dest[pos++]='r';}
  else if(c=='\t'){dest[pos++]='\\';dest[pos++]='t';}
  else if(c<32){pos+=(size_t)sprintf(dest+pos,"\\u%04x",c);}
  else dest[pos++]=(char)c;
 }
 dest[pos++]='"';dest[pos]=0;
}
static int infer(const char *binary,const char *checkpoint,const char *prompt,const char *bytes,const char *seed,char *output,size_t cap){
 int fds[2];if(pipe(fds)<0)return -1;
 pid_t pid=fork();
 if(pid<0){close(fds[0]);close(fds[1]);return -1;}
 if(pid==0){
  close(fds[0]);dup2(fds[1],STDOUT_FILENO);close(fds[1]);
  execl(binary,binary,"generate",checkpoint,prompt,bytes,seed,(char*)NULL);_exit(127);
 }
 close(fds[1]);size_t used=0;ssize_t n;
 while(used+1<cap&&(n=read(fds[0],output+used,cap-used-1))>0)used+=(size_t)n;
 close(fds[0]);output[used]=0;
 int status=0;if(waitpid(pid,&status,0)<0)return -1;
 return WIFEXITED(status)&&WEXITSTATUS(status)==0?0:-1;
}
static int integer(const char *s,long minimum,long maximum){
 if(!s||!*s)return 0;char *end;errno=0;long v=strtol(s,&end,10);
 return !errno&&end&&!*end&&v>=minimum&&v<=maximum;
}
static void handle(int fd,const char *binary,const char *checkpoint){
 char request[MAX_REQ+1];size_t total=0;ssize_t got;
 while(total<MAX_REQ){
  got=recv(fd,request+total,MAX_REQ-total,0);
  if(got<=0)return;
  total+=(size_t)got;request[total]=0;
  if(strstr(request,"\r\n\r\n"))break;
 }
 char *body=strstr(request,"\r\n\r\n");
 if(!body){response(fd,400,"{\"error\":\"invalid HTTP headers\"}");return;}
 size_t header_len=(size_t)(body-request)+4;body+=4;
 int is_get=!strncmp(request,"GET ",4),is_post=!strncmp(request,"POST ",5);
 if(is_get&&!strncmp(request,"GET /health ",12)){
  response(fd,200,"{\"status\":\"ready\",\"model\":\"micro_llm\"}");return;
 }
 if(!is_post||strncmp(request,"POST /generate ",15)){
  response(fd,404,"{\"error\":\"not found\"}");return;
 }
 char *cl=NULL;
 for(char *p=strstr(request,"\r\n")+2;p<body-2;){
  char *e=strstr(p,"\r\n");if(!e||e>=body)break;
  if(!strncasecmp(p,"Content-Length:",15)){cl=p+15;break;}
  p=e+2;
 }
 if(!cl){response(fd,400,"{\"error\":\"Content-Length required\"}");return;}
 while(*cl==' ')cl++;
 char digits[32]={0};size_t ci=0;
 while(*cl>='0'&&*cl<='9'&&ci+1<sizeof(digits))digits[ci++]=*cl++;
 if(!integer(digits,1,8192)||(*cl!='\r'&&*cl!=' ')){
  response(fd,413,"{\"error\":\"invalid request length\"}");return;
 }
 size_t length=(size_t)strtoul(digits,NULL,10);
 if(header_len+length>MAX_REQ){response(fd,413,"{\"error\":\"request too large\"}");return;}
 while(total<header_len+length){
  got=recv(fd,request+total,header_len+length-total,0);
  if(got<=0)return;total+=(size_t)got;
 }
 request[header_len+length]=0;
 char prompt[4096]={0},limit[16]="64",seed[16]="913";
 JField f[]={{prompt,sizeof(prompt),"prompt",0,0},{limit,sizeof(limit),"max_bytes",0,0},{seed,sizeof(seed),"seed",0,0}};
 if(jfields(body,f,3)||f[0].type!=1||!prompt[0]||strlen(prompt)>2048||
    !integer(limit,1,256)||!integer(seed,0,2147483647)){
  response(fd,400,"{\"error\":\"invalid generation request\"}");return;
 }
 char generated[8192];
 if(infer(binary,checkpoint,prompt,limit,seed,generated,sizeof(generated))){
  response(fd,503,"{\"error\":\"inference failed\"}");return;
 }
 size_t len=strlen(generated);while(len&&generated[len-1]=='\n')--len;
 char escaped_text[16384];escaped(escaped_text,sizeof(escaped_text),generated,len);
 char answer[16400];snprintf(answer,sizeof(answer),"{\"text\":%s}",escaped_text);
 response(fd,200,answer);
}
int main(int argc,char **argv){
 if(argc!=3&&argc!=4){fputs("usage: llm_service MICRO_LLM_BINARY CHECKPOINT [PORT]\n",stderr);return 2;}
 int port=argc==4?atoi(argv[3]):8765;
 if(port<1024||port>65535){fputs("invalid port\n",stderr);return 2;}
 struct stat st;if(stat(argv[1],&st)||stat(argv[2],&st)){perror("model/checkpoint");return 1;}
 signal(SIGPIPE,SIG_IGN);
 int server=socket(AF_INET,SOCK_STREAM,0);if(server<0){perror("socket");return 1;}
 int opt=1;setsockopt(server,SOL_SOCKET,SO_REUSEADDR,&opt,sizeof(opt));
 struct sockaddr_in addr={0};addr.sin_family=AF_INET;addr.sin_port=htons((uint16_t)port);
 inet_pton(AF_INET,"127.0.0.1",&addr.sin_addr);
 if(bind(server,(struct sockaddr*)&addr,sizeof(addr))||listen(server,8)){perror("bind/listen");close(server);return 1;}
 printf("LM HTTP listening on 127.0.0.1:%d\n",port);fflush(stdout);
 for(;;){
  int fd=accept(server,NULL,NULL);if(fd<0){if(errno==EINTR)continue;break;}
  struct timeval timeout={10,0};
  setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
  handle(fd,argv[1],argv[2]);close(fd);
 }
 close(server);return 1;
}
