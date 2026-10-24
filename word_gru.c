/* STN-Labz original CPU word-level GRU.
 * Standalone learning path for approved Digit train/validation text.
 * No external weights, models, services or dependencies.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#define H 32
#define VOC 2048
#define MAXTOK 500000
#define MAXWORD 48
#define EPOCHS 60
#define BPTT 24
#define LR 0.002f
#define WEIGHT_DECAY 0.0001f
typedef struct {char text[MAXWORD];int freq;} word;
static word words[VOC];
static int nw=3;
static unsigned long long rnd=20261021;
static unsigned long long rand64(void){rnd^=rnd>>12;rnd^=rnd<<25;rnd^=rnd>>27;return rnd*2685821657736338717ULL;}
static float urand(void){return (float)((rand64()>>40)/16777216.0f-.5f);}
typedef struct {
 float e[VOC][H],wz[H][H],wr[H][H],wn[H][H];
 float uz[H][H],ur[H][H],un[H][H];
 float bz[H],br[H],bn[H],out[H][VOC],bo[VOC];
} net;
typedef struct {
 int token,target,reset,supervised;
 float prev[H],z[H],r[H],n[H],h[H],prob[VOC];
} tick;
/* Count the full training vocabulary before selecting the fixed-size
 * model vocabulary. The counting table must not silently cap unique words.
 * It grows by powers of two with a bounded maximum allocation. Neither
 * the trained vocabulary width nor the checkpoint format changes. */
#define TERMS_INITIAL 32768u
#define TERMS_MAX 1048576u
typedef struct {char text[MAXWORD];unsigned int count;} term;
static term *counts;
static size_t term_slots;
static unsigned int distinct;
static unsigned hash_word(const char *s){
 unsigned h=2166136261u;
 while(*s){h=((h^(unsigned char)*s++)*16777619u);}
 return h;
}
static int grow_vocabulary_counts(void){
 size_t next=term_slots?term_slots*2u:TERMS_INITIAL;
 if(next>TERMS_MAX)return 0;
 term *expanded=calloc(next,sizeof(*expanded));
 if(!expanded)return 0;
 if(counts){
  for(size_t i=0;i<term_slots;i++){
   if(!counts[i].count)continue;
   size_t slot=(size_t)hash_word(counts[i].text)&(next-1u);
   while(expanded[slot].count)slot=(slot+1u)&(next-1u);
   expanded[slot]=counts[i];
  }
  free(counts);
 }
 counts=expanded;
 term_slots=next;
 return 1;
}
static int lookup(const char *p){
 for(int i=3;i<nw;i++)if(!strcmp(p,words[i].text))return i;
 return 2;
}
static int vocabulary(const char *p){
 if(!counts)return 0;
 for(;;){
  size_t slot=(size_t)hash_word(p)&(term_slots-1u);
  for(size_t i=0;i<term_slots;i++){
   term *t=&counts[(slot+i)&(term_slots-1u)];
   if(!t->count){
    if(term_slots<TERMS_MAX &&
       ((size_t)distinct+1u)*10u>=term_slots*7u){
     if(!grow_vocabulary_counts())return 0;
     break;
    }
    memcpy(t->text,p,strlen(p)+1);
    t->count=1;
    distinct++;
    return 1;
   }
   if(!strcmp(t->text,p)){
    if(t->count==UINT32_MAX)return 0;
    t->count++;
    return 1;
   }
  }
  if(term_slots==TERMS_MAX)return 0;
 }
}
static int word_order(const void *a,const void *b){
 const term *x=(const term*)a,*y=(const term*)b;
 if(x->count>y->count)return -1;
 if(x->count<y->count)return 1;
 return strcmp(x->text,y->text);
}
static int finish_vocabulary(void){
 qsort(counts,term_slots,sizeof(*counts),word_order);
 for(unsigned i=0;i<distinct&&nw<VOC;i++){
  if(!counts[i].count)break;
  strcpy(words[nw].text,counts[i].text);
  words[nw++].freq=(int)counts[i].count;
 }
 free(counts);counts=NULL;
 return nw>10;
}
static unsigned char *load(const char *path,size_t *n){
 FILE *f=fopen(path,"rb");if(!f)return NULL;
 if(fseek(f,0,SEEK_END)){fclose(f);return NULL;}
 long l=ftell(f);if(l<=0||l>16*1024*1024L){fclose(f);return NULL;}
 rewind(f);unsigned char *p=malloc((size_t)l+1);
 if(!p){fclose(f);return NULL;}
 if(fread(p,1,(size_t)l,f)!=(size_t)l){free(p);fclose(f);return NULL;}
 fclose(f);p[l]=0;*n=(size_t)l;return p;
}
static int *tokenize(const unsigned char *src,size_t n,int *out,int build){
 int *seq=malloc((n+2)*sizeof(int));if(!seq)return NULL;
 int count=0;char buf[MAXWORD];int len=0;
 for(size_t i=0;i<=n;i++){
  unsigned char c=i<n?src[i]:'\n';
  if((c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='\''){
   if(len<MAXWORD-1)buf[len++]=(char)tolower(c);
  }else{
   if(len){
    buf[len]=0;if(build){if(!vocabulary(buf)){free(seq);return NULL;}}
    else seq[count++]=lookup(buf);
    len=0;
   }
   if(!build && (c=='\n'||c=='.'||c=='?'||c=='!')){
    if(count && seq[count-1]!=1)seq[count++]=1;
   }
  }
 }
 *out=count;return seq;
}
static void init(net *m){
 memset(m,0,sizeof(*m));
 float *w=(float*)m;for(size_t i=0;i<sizeof(*m)/sizeof(float);i++)w[i]=urand()*0.12f;
}
static float sig(float x){return x>=0?1.f/(1.f+expf(-x)):expf(x)/(1.f+expf(x));}
static void forward(net *m,const float *prev,int tok,tick *t){
 t->token=tok;memcpy(t->prev,prev,H*sizeof(float));
 for(int j=0;j<H;j++){
  float az=m->bz[j],ar=m->br[j];
  for(int i=0;i<H;i++){az+=m->e[tok][i]*m->wz[i][j]+prev[i]*m->uz[i][j];ar+=m->e[tok][i]*m->wr[i][j]+prev[i]*m->ur[i][j];}
  t->z[j]=sig(az);t->r[j]=sig(ar);
 }
 for(int j=0;j<H;j++){
  float a=m->bn[j];
  for(int i=0;i<H;i++)a+=m->e[tok][i]*m->wn[i][j]+(prev[i]*t->r[i])*m->un[i][j];
  t->n[j]=tanhf(a);t->h[j]=t->z[j]*prev[j]+(1-t->z[j])*t->n[j];
 }
 float max=-1e30f,sum=0;
 for(int k=0;k<nw;k++){
  float a=m->bo[k];for(int j=0;j<H;j++)a+=t->h[j]*m->out[j][k];
  t->prob[k]=a;if(a>max)max=a;
 }
 for(int k=0;k<nw;k++){t->prob[k]=expf(t->prob[k]-max);sum+=t->prob[k];}
 for(int k=0;k<nw;k++)t->prob[k]/=sum;
}
static float clip(float x){return x>5?5:x< -5?-5:x;}
static net *moment1,*moment2;
static unsigned long adam_steps;
static int optimizer_init(void){
 moment1=calloc(1,sizeof(*moment1));moment2=calloc(1,sizeof(*moment2));
 return moment1&&moment2;
}
static void optimize(net *m,const net *g,int count){
 float *w=(float*)m,*v=(float*)moment1,*q=(float*)moment2;
 const float *grad=(const float*)g;
 ++adam_steps;
 const double corr1=1.0-pow(0.9,(double)adam_steps);
 const double corr2=1.0-pow(0.999,(double)adam_steps);
 for(size_t i=0;i<sizeof(*m)/sizeof(float);i++){
  float d=clip(grad[i]/(float)count);
  v[i]=0.9f*v[i]+0.1f*d;
  q[i]=0.999f*q[i]+0.001f*d*d;
  double a=(double)v[i]/corr1,b=(double)q[i]/corr2;
  w[i]-=(float)(LR*a/(sqrt(b)+1e-8));
  w[i]*=(1.0f-LR*WEIGHT_DECAY);
 }
}
static double learn(net *m,tick *ts,int count){
 net *g=calloc(1,sizeof(*g));if(!g)return -1;
 float future[H]={0};double loss=0;
 for(int z=count-1;z>=0;z--){
  tick *t=&ts[z];float dh[H],dn[H],dz[H],dr[H],back[H],de[H]={0};
  float err[VOC];for(int k=0;k<nw;k++){
   err[k]=t->supervised?(t->prob[k]-(k==t->target)):0;
   g->bo[k]+=err[k];
  }
  if(t->supervised)loss-=log(fmax((double)t->prob[t->target],1e-25));
  for(int j=0;j<H;j++){
   dh[j]=future[j];for(int k=0;k<nw;k++){g->out[j][k]+=t->h[j]*err[k];dh[j]+=m->out[j][k]*err[k];}
   dn[j]=dh[j]*(1-t->z[j])*(1-t->n[j]*t->n[j]);
   dz[j]=dh[j]*(t->prev[j]-t->n[j])*t->z[j]*(1-t->z[j]);
  }
  for(int i=0;i<H;i++){
   float a=0;for(int j=0;j<H;j++)a+=m->un[i][j]*dn[j];
   dr[i]=a*t->prev[i]*t->r[i]*(1-t->r[i]);
  }
  for(int i=0;i<H;i++){
   float a=0;for(int j=0;j<H;j++)a+=m->un[i][j]*dn[j];
   back[i]=dh[i]*t->z[i]+a*t->r[i];
   for(int j=0;j<H;j++)back[i]+=m->uz[i][j]*dz[j]+m->ur[i][j]*dr[j];
  }
  for(int i=0;i<H;i++)for(int j=0;j<H;j++){
   g->wz[i][j]+=m->e[t->token][i]*dz[j];g->wr[i][j]+=m->e[t->token][i]*dr[j];
   g->wn[i][j]+=m->e[t->token][i]*dn[j];
   g->uz[i][j]+=t->prev[i]*dz[j];g->ur[i][j]+=t->prev[i]*dr[j];
   g->un[i][j]+=t->prev[i]*t->r[i]*dn[j];
   de[i]+=m->wz[i][j]*dz[j]+m->wr[i][j]*dr[j]+m->wn[i][j]*dn[j];
  }
  for(int j=0;j<H;j++){g->bz[j]+=dz[j];g->br[j]+=dr[j];g->bn[j]+=dn[j];g->e[t->token][j]+=de[j];}
  memcpy(future,back,sizeof(future));
  if(t->reset)memset(future,0,sizeof(future));
 }
 optimize(m,g,count);
 free(g);return loss;
}
/* Tokenize a verified Q/A record as one sequence. Unlike ordinary sentence
 * tokenization, '?' is not an EOS; the answer sees the question's state.
 * Only answer tokens and EOS receive a predictive loss. */
static int dialogue_tokens(const char *line,int *sequence,unsigned char *mask,
                           int cap){
 const char *answer=strstr(line," Answer: ");
 if(strncmp(line,"Question: ",10)||!answer||answer==line+10)return -1;
 if(!answer[9]||strstr(answer+9," Answer: "))return -1;
 int count=0,seen_answer=0,previous=0,answer_words=0;
 char wordbuf[MAXWORD];int length=0;
 for(const char *p=line;;p++){
  unsigned char c=(unsigned char)*p;
  int letter=(c>='A'&&c<='Z')||(c>='a'&&c<='z')||
             (c>='0'&&c<='9')||c=='\'';
  if(letter){
   if(length<MAXWORD-1)wordbuf[length++]=(char)tolower(c);
  }else{
   if(length){
    wordbuf[length]=0;
    if(count>=cap-2)return -1;
    sequence[count]=lookup(wordbuf);
    mask[count]=(unsigned char)(seen_answer&&p>answer+9);
    if(mask[count])answer_words++;
    if(!strcmp(wordbuf,"answer")&&p>answer&&p<=answer+7)
     seen_answer=1;
    previous=sequence[count];count++;length=0;
   }
   if(c=='\n'||c==0){
    if(!seen_answer||!answer_words||count>=cap)return -1;
    if(previous!=1){sequence[count]=1;mask[count]=1;count++;}
    return count;
   }
  }
 }
}
static int dialogue_train(net *m,const unsigned char *source,size_t bytes){
 /* One supervised pass over the genuine approved dialogue pairs.
  * There is no artificial replay, duplication, or invented target. */
 char line[8192];int processed=0,supervised=0;
 const unsigned char *p=source,*end=source+bytes;
 while(p<end){
  const unsigned char *e=memchr(p,'\n',(size_t)(end-p));
  if(!e||(size_t)(e-p)>=sizeof line-1)return -1;
  size_t size=(size_t)(e-p);
  memcpy(line,p,size);line[size]='\n';line[size+1]=0;p=e+1;
  int tokens[4096];unsigned char mask[4096]={0};
  int count=dialogue_tokens(line,tokens,mask,4096);
  if(count<=0)return -1;
  float h[H]={0};int prev=0;
  for(int pos=0;pos<count;){
   tick ts[BPTT];int n=0,targets=0;
   while(n<BPTT&&pos<count){
    int target=tokens[pos];
    forward(m,h,prev,&ts[n]);
    ts[n].target=target;ts[n].reset=prev==0;
    ts[n].supervised=mask[pos];
    targets+=mask[pos];
    memcpy(h,ts[n].h,sizeof h);prev=target;n++;pos++;
    if(target==1){memset(h,0,sizeof h);prev=0;}
   }
   if(targets){
    if(learn(m,ts,n)<0)return -1;
    supervised+=targets;
   }
  }
  processed++;
 }
 printf("dialogue_pairs_supervised=%d answer_tokens_supervised=%d\n",
        processed,supervised);
 return processed;
}
static double evaluate(net *m,const int *seq,int count){
 float h[H]={0};tick t;int prev=0;double loss=0;int used=0;
 for(int i=0;i<count;i++){
  forward(m,h,prev,&t);loss-=log(fmax((double)t.prob[seq[i]],1e-25));used++;
  memcpy(h,t.h,sizeof(h));prev=seq[i];
  if(prev==1){memset(h,0,sizeof(h));prev=0;}
 }
 return used?loss/used:1e9;
}
static int save(const char *path,net *m){
 FILE *f=fopen(path,"wb");if(!f)return 0;
 const uint32_t magic=0x57524731;int ok=fwrite(&magic,4,1,f)==1&&fwrite(&nw,sizeof(nw),1,f)==1&&fwrite(words,sizeof(words),1,f)==1&&fwrite(m,sizeof(*m),1,f)==1;
 if(fclose(f))ok=0;
 return ok;
}
static int restore(const char *path,net *m){
 FILE *f=fopen(path,"rb");if(!f)return 0;uint32_t magic=0;
 int ok=fread(&magic,4,1,f)==1&&magic==0x57524731&&fread(&nw,sizeof(nw),1,f)==1&&nw>=10&&nw<=VOC&&fread(words,sizeof(words),1,f)==1&&fread(m,sizeof(*m),1,f)==1&&fgetc(f)==EOF;
 fclose(f);return ok;
}
static const int *training_tokens;
static int training_count;
static unsigned int *transitions;
static void build_transitions(const int *tokens,int count){
 transitions=calloc((size_t)VOC*VOC,sizeof(*transitions));
 if(!transitions)return;
 int previous=0;
 for(int i=0;i<count;i++){
  int next=tokens[i];
  if(previous>=0&&previous<nw&&next>=1&&next<nw)
   transitions[(size_t)previous*VOC+next]++;
  previous=next==1?0:next;
 }
}
static int produce(net *m,const char *path){
 FILE *f=fopen(path,"wb");if(!f||!transitions){if(f)fclose(f);return 0;}
 tick t;float h[H]={0};int prev=0,last=-1,words_out=0,penultimate=0;
 /* Word-order transitions are learned only from the training split.
  * The GRU scores each permitted next word. Rejecting unattested
  * adjacent words prevents the decoder inventing impossible local syntax.
  * This is constrained generation, not a grammaticality guarantee. */
 for(int i=0;i<32;i++){
  forward(m,h,prev,&t);
  double sum=0,weights[VOC]={0};
  for(int k=1;k<nw;k++){
   unsigned n=transitions[(size_t)prev*VOC+k];
   if(!n||k==2)continue;
   if(words_out>0){
    n=0;
    for(int j=1;j<training_count;j++){
     int a=training_tokens[j-1],b=training_tokens[j];
     int before=j>=2?(training_tokens[j-2]==1?0:training_tokens[j-2]):0;
     if(before==penultimate&&a==prev&&b==k)n++;
     /* Crossing sentence endings cannot supply a phrase continuation. */
    }
   }
   if(!n)continue;
   if(k==1&&words_out<5)continue;
   double p=pow(fmax(t.prob[k],1e-15f),1.0/0.68);
   p*=sqrt((double)n);
   if(k==last)p*=0.02;
   weights[k]=p;sum+=p;
  }
  if(!(sum>0))break;
  double pick=((rand64()>>40)/16777216.0)*sum;int choice=-1;
  for(int k=1;k<nw;k++)if(weights[k]>0){
   choice=k;pick-=weights[k];if(pick<=0)break;
  }
  if(choice<0)break;
  if(choice==1)break;
  if(words_out)fputc(' ',f);
  if(!words_out){
   const char *p=words[choice].text;
   if(*p){fputc(toupper((unsigned char)*p),f);fputs(p+1,f);}
  }else fputs(words[choice].text,f);
  words_out++;last=choice;
  memcpy(h,t.h,sizeof(h));penultimate=prev;prev=choice;
 }
 fputs(".\n",f);
 return fclose(f)==0&&words_out>=6;
}
/* Standalone Linux/Windows inference for the versioned word GRU checkpoint.
 * Sampling does not require the training corpus or the training-only trigram table.
 * Output is model-generated text, not a canned response or corpus lookup. */
static int sample_checkpoint(int argc,char **argv){
 if(argc!=8){fputs("usage: word_gru sample CHECKPOINT PROMPT WORDS SEED TEMPERATURE PENALTY\\n",stderr);return 2;}
 char *end=NULL;
 long limit=strtol(argv[4],&end,10);
 if(!argv[4][0]||*end||limit<1||limit>128)return 2;
 unsigned long long seed=strtoull(argv[5],&end,10);
 if(!argv[5][0]||*end||seed==0)return 2;
 float temperature=strtof(argv[6],&end);
 if(!argv[6][0]||*end||!isfinite(temperature)||temperature<0.25f||temperature>2.0f)return 2;
 float penalty=strtof(argv[7],&end);
 if(!argv[7][0]||*end||!isfinite(penalty)||penalty<0||penalty>2)return 2;
 net *m=malloc(sizeof(*m));
 if(!m)return 1;
 if(!restore(argv[2],m)){fputs("invalid or incompatible word GRU checkpoint\\n",stderr);free(m);return 1;}
 rnd=seed;
 float h[H]={0};tick t;int prev=lookup("question");
 const unsigned char *p=(const unsigned char*)argv[3];char token[MAXWORD];int len=0;
 /* Mirror the supervised Q/A context at inference: Question [prompt]
  * Answer [generation]. This only conditions the recurrent state and
  * never looks up or copies any saved answer. */
 for(size_t i=0;;i++){
  unsigned char c=p[i];int letter=(c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='\'';
  if(letter){if(len<MAXWORD-1)token[len++]=(char)tolower(c);}
  else if(len){
   token[len]=0;
   forward(m,h,prev,&t);memcpy(h,t.h,sizeof(h));
   prev=lookup(token);len=0;
  }
  if(!c)break;
 }
 /* Advance through the final question word before the answer delimiter. */
 forward(m,h,prev,&t);memcpy(h,t.h,sizeof h);
 prev=lookup("answer");
 int used[VOC]={0};int output=0;
 for(int step=0;step<limit;step++){
  forward(m,h,prev,&t);
  double sum=0;double weight[VOC];
  for(int k=0;k<nw;k++){
   if(k==0||k==2||(k==1&&output<3)){weight[k]=0;continue;}
   double w=pow(fmax((double)t.prob[k],1e-20),1.0/(double)temperature);
   if(k>2&&used[k])w/=1.0+(double)penalty*used[k];
   weight[k]=w;sum+=w;
  }
  if(!(sum>0))break;
  double draw=((rand64()>>40)/16777216.0)*sum;
  int chosen=1;
  for(int k=0;k<nw;k++){draw-=weight[k];if(draw<=0){chosen=k;break;}}
  if(chosen==1)break;
  if(output)putchar(' ');
  const char *w=words[chosen].text;
  if(!output&&*w){putchar(toupper((unsigned char)*w));fputs(w+1,stdout);}
  else fputs(w,stdout);
  output++;used[chosen]++;
  memcpy(h,t.h,sizeof(h));prev=chosen;
 }
 puts(".");
 free(m);
 return output?0:1;
}
/* Deterministic, data-free fixture for the new supervised record boundary. */
static int test_dialogue(void){
 int seq[64]={0};unsigned char mask[64]={0};
 int n=dialogue_tokens("Question: Hello? Answer: Hi there.\n",
                      seq,mask,64);
 if(n!=6||mask[0]||mask[1]||mask[2]||
    mask[3]!=1||mask[4]!=1||mask[5]!=1||seq[5]!=1){
  fputs("dialogue_tokenization=FAIL\n",stderr);return 1;
 }
 if(dialogue_tokens("Question: no answer here\n",seq,mask,64)>=0 ||
    dialogue_tokens("Question: x Answer: \n",seq,mask,64)>=0){
  fputs("dialogue_negative_tests=FAIL\n",stderr);return 1;
 }
 puts("dialogue_tokenization=PASS");
 puts("dialogue_negative_tests=PASS");
 return 0;
}
int main(int argc,char **argv){
 if(argc>1&&!strcmp(argv[1],"test-dialogue"))return test_dialogue();
 if(argc>1&&!strcmp(argv[1],"sample"))return sample_checkpoint(argc,argv);
 if((argc!=6&&argc!=7)||strcmp(argv[1],"train")){
  fputs("usage: word_gru train TRAIN VALID MODEL SAMPLE [DIALOGUE]\n",stderr);return 2;
 }
 size_t tn=0,vn=0;unsigned char *tr=load(argv[2],&tn),*va=load(argv[3],&vn);
 if(!tr||!va){fputs("training corpus unavailable\n",stderr);free(tr);free(va);return 1;}
 strcpy(words[0].text,"<bos>");strcpy(words[1].text,"<eos>");strcpy(words[2].text,"<unk>");
 if(!grow_vocabulary_counts()){free(tr);free(va);return 1;}
 int count=0;int *scratch=tokenize(tr,tn,&count,1);
 if(!scratch||!finish_vocabulary()){fputs("vocabulary construction failed\n",stderr);free(scratch);free(counts);free(tr);free(va);return 1;}
 free(scratch);
 printf("vocabulary_selected=%d distinct_training_words=%u counting_slots=%zu\n",nw-3,distinct,term_slots);
 int nt=0,nv=0;int *train=tokenize(tr,tn,&nt,0),*valid=tokenize(va,vn,&nv,0);
 free(tr);free(va);
 int unknown_train=0,unknown_valid=0;
 if(train)for(int i=0;i<nt;i++)if(train[i]==2)unknown_train++;
 if(valid)for(int i=0;i<nv;i++)if(valid[i]==2)unknown_valid++;
 printf("train_unknown=%d/%d validation_unknown=%d/%d\n",unknown_train,nt,unknown_valid,nv);
 if(!train||!valid||nt<100||nv<10){fputs("insufficient tokenized corpus\n",stderr);free(train);free(valid);return 1;}
 net *m=malloc(sizeof(*m)),*best=malloc(sizeof(*m));
 if(!m||!best){free(m);free(best);free(train);free(valid);return 1;}
 if(!optimizer_init()){free(m);free(best);free(train);free(valid);free(moment1);free(moment2);return 1;}
 init(m);double best_loss=1e9;int bad_epochs=0;
 for(int epoch=0;epoch<EPOCHS;epoch++){
  double total=0;int pos=0;float h[H]={0};int prev=0;
  while(pos<nt){
   tick ts[BPTT];int n=0;
   while(n<BPTT&&pos<nt){
    int target=train[pos++];forward(m,h,prev,&ts[n]);
    ts[n].target=target;ts[n].reset=prev==0;ts[n].supervised=1;
    memcpy(h,ts[n].h,sizeof(h));prev=target;n++;
    if(target==1){memset(h,0,sizeof(h));prev=0;}
   }
   double l=learn(m,ts,n);if(l<0){free(m);free(best);free(train);free(valid);return 1;}total+=l;
  }
  double vl=evaluate(m,valid,nv);
  printf("epoch=%d train_loss=%.5f valid_loss=%.5f\n",epoch+1,total/nt,vl);
  if(vl<best_loss-0.0005){best_loss=vl;memcpy(best,m,sizeof(*m));bad_epochs=0;}
  else if(++bad_epochs>=5){printf("early_stop_epoch=%d\n",epoch+1);break;}
 }
 if(argc==7){
  size_t dn=0;unsigned char *dialogue=load(argv[6],&dn);
  if(!dialogue){
   fputs("dialogue corpus missing, empty or exceeds 16 MiB\n",stderr);
   free(m);free(best);free(moment1);free(moment2);
   free(train);free(valid);return 1;
  }
  /* The optimizer history belongs to the final epoch, not the best
   * validation checkpoint. Reset its moments for this separate objective. */
  memset(moment1,0,sizeof(*moment1));
  memset(moment2,0,sizeof(*moment2));
  adam_steps=0;
  int pairs=dialogue_train(best,dialogue,dn);
  free(dialogue);
  if(pairs<=0){
   fputs("dialogue corpus malformed or no supervised pairs\n",stderr);
   free(m);free(best);free(moment1);free(moment2);
   free(train);free(valid);return 1;
  }
 }else{
  puts("dialogue_pairs_supervised=0");
 }
 training_tokens=train;training_count=nt;
 build_transitions(train,nt);
 int ok=transitions&&save(argv[4],best)&&produce(best,argv[5]);
 printf("tokens=%d vocabulary=%d best_validation_loss=%.5f\n",nt,nw,best_loss);
 free(m);free(best);free(moment1);free(moment2);free(transitions);free(train);free(valid);return ok?0:1;
}