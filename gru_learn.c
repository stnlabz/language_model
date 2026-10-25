/* gru_learn: native C approved JSONL -> temporary plain-text corpus -> native LM trainer.
 * Model is a causal Transformer. No GRU is used.
 * Linux/POSIX, no Python. Never modifies approved source.
 * Blank-line record boundaries prevent adjacent unrelated lessons from
 * being presented as one uninterrupted text sequence.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <limits.h>
#include "json_fields.h"
#define LINE_MAX_BYTES 131072
static int starts(const char *s,const char *prefix){return strncmp(s,prefix,strlen(prefix))==0;}
static int convert(const char *input,FILE *out){
 FILE *f=fopen(input,"rb");if(!f){perror(input);return -1;}
 char *line=malloc(LINE_MAX_BYTES);
 char *question=malloc(65536),*answer=malloc(65536),*statement=malloc(65536);
 if(!line||!question||!answer||!statement){fclose(f);free(line);free(question);free(answer);free(statement);return -1;}
 size_t nline=0,accepted=0;
 int ok=1;
 while(fgets(line,LINE_MAX_BYTES,f)){
  ++nline;
  size_t len=strlen(line);
  if(len==LINE_MAX_BYTES-1&&line[len-1]!='\n'){fprintf(stderr,"oversize record at line %zu\n",nline);ok=0;break;}
  if(len==0||strspn(line," \r\n\t")==len)continue;
  char approved[8]={0},type[64]={0};
  JField fields[]={
   {approved,sizeof(approved),"approved",0,0},
   {type,sizeof(type),"type",0,0},
   {question,65536,"input",0,0},
   {answer,65536,"response",0,0},
   {statement,65536,"text",0,0}
  };
  if(jfields(line,fields,5)){fprintf(stderr,"invalid JSON or unsupported record at line %zu\n",nline);ok=0;break;}
  if(fields[0].type!=2)continue;
  if(!strcmp(type,"statement")&&fields[4].type==1&&statement[0]){
   if(fprintf(out,"%s\n\n",statement)<0){ok=0;break;}
   ++accepted;continue;
  }
  if(fields[3].type!=1||!answer[0]||fields[2].type!=1)continue;
  if(starts(question,"Digit Corpus record ")){
   if(fprintf(out,"%s\n\n",answer)<0){ok=0;break;}
   ++accepted;
  }else if(!starts(question,"Digit Context record ")&&
           !starts(question,"Digit Observation record ")){
   if(fprintf(out,"Question: %s\nAnswer: %s\n\n",question,answer)<0){ok=0;break;}
   ++accepted;
  }
 }
 if(ferror(f))ok=0;
 fclose(f);free(line);free(question);free(answer);free(statement);
 if(accepted<5){fprintf(stderr,"only %zu approved usable records\n",accepted);ok=0;}
 if(ok)fprintf(stderr,"approved_training_records=%zu\n",accepted);
 return ok?0:-1;
}
int main(int argc,char **argv){
 if(argc!=5){fputs("usage: gru_learn APPROVED_JSONL NEW_CHECKPOINT STEPS MICRO_LLM_BINARY\n",stderr);return 2;}
 char *end=NULL;errno=0;long steps=strtol(argv[3],&end,10);
 if(errno||!end||*end||steps<1||steps>200000){fputs("invalid step count\n",stderr);return 2;}
 struct stat st;if(stat(argv[2],&st)==0){fputs("checkpoint exists; refusing overwrite\n",stderr);return 2;}
 if(errno!=ENOENT){perror(argv[2]);return 1;}
 char temp[]="/tmp/stn-gru-learn-XXXXXX";int fd=mkstemp(temp);
 if(fd<0){perror("mkstemp");return 1;}
 FILE *corpus=fdopen(fd,"wb");if(!corpus){close(fd);unlink(temp);return 1;}
 int valid=convert(argv[1],corpus);
 if(fflush(corpus))valid=-1;
 if(fclose(corpus))valid=-1;
 if(valid){unlink(temp);return 1;}
 pid_t pid=fork();
 if(pid<0){perror("fork");unlink(temp);return 1;}
 if(pid==0){
  execl(argv[4],argv[4],"train",temp,argv[2],argv[3],(char*)NULL);
  perror("exec LM");_exit(127);
 }
 int status=0;
 if(waitpid(pid,&status,0)<0){perror("waitpid");unlink(temp);return 1;}
 unlink(temp);
 if(!WIFEXITED(status)||WEXITSTATUS(status)!=0){
  fputs("LM training failed; checkpoint not qualified\n",stderr);return 1;
 }
 printf("learning_finished checkpoint=%s\n",argv[2]);return 0;
}
