/* Frozen-checkpoint position/content diagnostics. No optimization or decoding filters. */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main
static void probe(const char *kind,const char *subject,const char *prompt){
 const char *answer=!strcmp(subject,"engineer")?" An engineer designs systems.":!strcmp(subject,"mechanic")?" A mechanic repairs machines.":" A gardener grows plants.";
 const char *wrong=(!strcmp(subject,"engineer")||!strcmp(subject,"mechanic"))?" A gardener grows plants.":" An engineer designs systems.";
 unsigned char buf[512];size_t plen=strlen(prompt),alen=strlen(answer);Cache *c=malloc(sizeof(*c));
 double loss[2]={0,0};int teacher_correct=0;
 for(int option=0;option<2;option++){
  const char *target=option?wrong:answer;size_t length=strlen(target);
  memcpy(buf,prompt,plen);memcpy(buf+plen,target,length);
  for(size_t j=0;j<length;j++){
   size_t used=plen+j;int n=used<T?(int)used:T;forward(buf+used-n,n,-1,c);
   loss[option]-=log(fmax(c->prob[n-1][(unsigned char)target[j]],1e-30));
   int best=0;for(int b=1;b<V;b++)if(c->prob[n-1][b]>c->prob[n-1][best])best=b;
   if(!option&&best==(unsigned char)target[j])teacher_correct++;
  }
  loss[option]/=length;
 }
 memcpy(buf,prompt,plen);int exact=1;
 for(size_t j=0;j<alen;j++){
  size_t used=plen+j;int n=used<T?(int)used:T;forward(buf+used-n,n,-1,c);
  int best=0;for(int b=1;b<V;b++)if(c->prob[n-1][b]>c->prob[n-1][best])best=b;
  buf[used]=(unsigned char)best;if(best!=(unsigned char)answer[j])exact=0;
 }
 printf("%s,%s,%zu,%d,%d,%zu,%.6f,%.6f,\"",kind,subject,plen,exact,teacher_correct,alen,loss[0],loss[1]);
 for(size_t j=0;j<alen;j++){unsigned char b=buf[plen+j];if(b=='"')printf("\"\"");else if(b>=32&&b<127)putchar(b);else printf("\\x%02x",b);}
 puts("\"");free(c);
}
int main(int argc,char **argv){
 if(argc!=2)return 2;
 init();if(!restore(argv[1]))return 1;
 puts("variant,subject,prefix_bytes,exact,teacher_correct,answer_bytes,correct_nll,wrong_nll,generated");
 const char *subjects[]={"engineer","gardener"};
 for(int s=0;s<2;s++){
  const char *article=s?"a":"an",*job=subjects[s];char prompt[400],base[300],label[80];
  snprintf(base,sizeof(base),"Question: What is %s %s?\nAnswer:",article,job);
  probe("trained_short",job,base);
  snprintf(prompt,sizeof(prompt),"Question: Tell me what %s %s does.\nAnswer:",article,job);probe("reference_tell",job,prompt);
  snprintf(prompt,sizeof(prompt),"Question: What work does %s %s perform?\nAnswer:",article,job);probe("reference_work",job,prompt);
  snprintf(prompt,sizeof(prompt),"Question: Describe %s %s's job.\nAnswer:",article,job);probe("reference_describe",job,prompt);
  snprintf(prompt,sizeof(prompt),"Question: Explain the job of %s %s.\nAnswer:",article,job);probe("heldout_explain",job,prompt);
  snprintf(prompt,sizeof(prompt),"Question: What does %s %s do for work?\nAnswer:",article,job);probe("heldout_work",job,prompt);
  for(int k=1;k<=12;k++){
   memset(prompt,' ',(size_t)k);strcpy(prompt+k,base);snprintf(label,sizeof(label),"leftpad_%02d",k);probe(label,job,prompt);
  }
  strcpy(prompt,base);*strchr(prompt,'\n')=' ';probe("newline_to_space",job,prompt);
  strcpy(prompt,base);*strchr(prompt,'?')='.';probe("question_to_period",job,prompt);
  strcpy(prompt,base);prompt[0]='q';probe("lowercase_question",job,prompt);
  /* Same prefix length and syntactic frame, content identity removed. */
  strcpy(prompt,base);char *at=strstr(prompt,job);memset(at,'x',strlen(job));probe("subject_erased",job,prompt);
  /* Retain job word and all byte positions; remove other question wording. */
  strcpy(prompt,base);at=strstr(prompt,job);char *end=strchr(prompt,'\n');
  for(char *p=prompt+10;p<end;p++)if(!(p>=at&&p<at+strlen(job))&&*p!=' ')*p='x';
  probe("wording_erased",job,prompt);
  strcpy(prompt,base);at=strstr(prompt,job);memcpy(at,subjects[1-s],8);
  probe("subject_swapped",subjects[1-s],prompt);
  /* Equal-length alternative wording, preserving answer position exactly. */
  snprintf(prompt,sizeof(prompt),"Question: Name me %s %s.\nAnswer:",article,job);probe("unseen_same_length",job,prompt);
 }
 probe("same_length_trained","mechanic","Question: What is a mechanic?\nAnswer:");
 free(model);return 0;
}
