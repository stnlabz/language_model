#define _POSIX_C_SOURCE 200809L
/* STN-Labz micro Transformer: original C11 CPU-only causal self-attention LM.
 * One decoder block, learned token/position embeddings, causal attention,
 * tanh feed-forward, tied output embedding. Analytic gradients, Adam.
 * Byte vocabulary avoids unknown tokens. No pretrained weights or services.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <dirent.h>
#include <sys/stat.h>
#include <limits.h>
#define D 48
#define F 96
#define T 96
#define V 256
#define NPARAM (V*D+T*D+4*D*D+D*F+F*D+F+D)
#define MAGIC 0x4d544c53u
#define VERSION 4u
#define AT(a,b,stride) ((a)*(stride)+(b))
static uint64_t seed=20261024;
static uint64_t rng(void){seed^=seed>>12;seed^=seed<<25;seed^=seed>>27;return seed*2685821657736338717ULL;}
static float rnd(void){return (float)((rng()>>40)/16777216.0-0.5);}
typedef struct {
 float emb[V][D],pos[T][D],q[D][D],k[D][D],v[D][D],o[D][D];
 float f1[D][F],f2[F][D],b1[F],b2[D];
} Model;
typedef struct {
 float x[T][D],q[T][D],k[T][D],v[T][D],att[T][T],mix[T][D];
 float h[T][D],ff[T][F],y[T][D],prob[T][V];
} Cache;
static float *grad,*moment,*variance;
static unsigned char *data;
static size_t data_len;
/* Answer-focused supervision starts at the token that predicts the first
 * answer byte (the position immediately preceding the answer). */
static int supervision_start=0;
static Model *model;
/* Read approved .txt lessons in deterministic lexical order. No lesson is
 * rewritten or converted on disk. Limit input to the existing 50MB budget. */
#define MAX_CORPUS_BYTES 50000000UL
#define MAX_LESSONS 4096
static size_t lesson_count=0,lesson_bytes=0;
static int lesson_name(const char *name){
 size_t n=strlen(name);
 return n>4 && strcmp(name+n-4,".txt")==0 && name[0]!='.';
}
static int cmp_names(const void *a,const void *b){
 return strcmp(*(const char *const*)a,*(const char *const*)b);
}
static int load_data(const char *path){
 struct stat st;
 if(lstat(path,&st)!=0){perror(path);return 0;}
 if(S_ISREG(st.st_mode)){
  FILE *f=fopen(path,"rb");if(!f){perror(path);return 0;}
  if(st.st_size<1024||st.st_size>(long)MAX_CORPUS_BYTES){
   fclose(f);fputs("corpus must be 1KiB..50MB\n",stderr);return 0;
  }
  data_len=(size_t)st.st_size;
  data=malloc(data_len);
  if(!data){fclose(f);return 0;}
  int ok=fread(data,1,data_len,f)==data_len && fgetc(f)==EOF;
  if(fclose(f))ok=0;
  if(!ok){free(data);data=NULL;return 0;}
  lesson_count=1;lesson_bytes=data_len;
  printf("corpus_source=file files=1 bytes=%zu\n",lesson_bytes);
  return 1;
 }
 if(!S_ISDIR(st.st_mode)){
  fputs("corpus path must be a regular file or directory\n",stderr);return 0;
 }
 DIR *dir=opendir(path);
 if(!dir){perror(path);return 0;}
 char *names[MAX_LESSONS];size_t count=0;
 struct dirent *ent;int ok=1;
 while((ent=readdir(dir))!=NULL){
  if(!lesson_name(ent->d_name))continue;
  if(count==MAX_LESSONS){fputs("too many lesson files\n",stderr);ok=0;break;}
  names[count]=malloc(strlen(ent->d_name)+1);
  if(!names[count]){ok=0;break;}
  strcpy(names[count],ent->d_name);count++;
 }
 if(closedir(dir))ok=0;
 if(ok&&count==0){fputs("no .txt lessons found\n",stderr);ok=0;}
 if(ok)qsort(names,count,sizeof(names[0]),cmp_names);
 size_t used=0,raw=0;
 if(ok){data=malloc(MAX_CORPUS_BYTES);if(!data)ok=0;}
 for(size_t i=0;ok&&i<count;i++){
  size_t pn=strlen(path),nn=strlen(names[i]);
  if(pn>SIZE_MAX-nn-2){ok=0;break;}
  char *full=malloc(pn+nn+2);
  if(!full){ok=0;break;}
  memcpy(full,path,pn);full[pn]='/';memcpy(full+pn+1,names[i],nn+1);
  struct stat item;
  if(lstat(full,&item)!=0 || !S_ISREG(item.st_mode) ||
     used>MAX_CORPUS_BYTES-2 || item.st_size<1 ||
     (uintmax_t)item.st_size>MAX_CORPUS_BYTES-used-2){
   fprintf(stderr,"invalid or oversized lesson: %s\n",full);
   free(full);ok=0;break;
  }
  FILE *f=fopen(full,"rb");
  if(!f){perror(full);free(full);ok=0;break;}
  size_t bytes=(size_t)item.st_size;
  if(fread(data+used,1,bytes,f)!=bytes||fgetc(f)!=EOF)ok=0;
  if(fclose(f))ok=0;
  if(!ok){fprintf(stderr,"could not read lesson: %s\n",full);free(full);break;}
  /* Verify stable content for this read; do not silently truncate files. */
  struct stat after;
  if(stat(full,&after)!=0 || after.st_size!=item.st_size){
   fprintf(stderr,"lesson changed during inspection: %s\n",full);
   free(full);ok=0;break;
  }
  free(full);
  used+=bytes;raw+=bytes;
  if(data[used-1]!='\n')data[used++]='\n';
  data[used++]='\n';
 }
 for(size_t i=0;i<count;i++)free(names[i]);
 if(!ok){free(data);data=NULL;return 0;}
 data_len=used;lesson_count=count;lesson_bytes=raw;
 printf("corpus_source=directory files=%zu bytes=%zu training_bytes=%zu\n",
        lesson_count,lesson_bytes,data_len);
 if(data_len<1024){free(data);data=NULL;fputs("corpus must be at least 1KiB\n",stderr);return 0;}
 return 1;
}
/* Initialize each parameter family at an appropriate scale.
 * Uniform +/-0.025 for every tensor made query/key scores effectively
 * zero at startup, leaving attention nearly uniform and gradients weak.
 * Embedding and projection scales are separated; biases start at zero.
 * STN-Labz: model-initialization correction, no external weights. */
static void init(void){
 model=calloc(1,sizeof(*model));if(!model)exit(2);
 for(int b=0;b<V;b++)for(int i=0;i<D;i++)
  model->emb[b][i]=rnd()*0.8f;
 for(int t=0;t<T;t++)for(int i=0;i<D;i++)
  model->pos[t][i]=rnd()*0.08f;
 for(int i=0;i<D;i++)for(int j=0;j<D;j++){
  model->q[i][j]=rnd()*0.36f;
  model->k[i][j]=rnd()*0.36f;
  model->v[i][j]=rnd()*0.36f;
  model->o[i][j]=rnd()*0.36f;
 }
 for(int i=0;i<D;i++)for(int j=0;j<F;j++)
  model->f1[i][j]=rnd()*0.28f;
 for(int i=0;i<F;i++)for(int j=0;j<D;j++)
  model->f2[i][j]=rnd()*0.28f;
}
static void matvec(const float *x,const float *w,float *y,int rows,int cols){
 for(int j=0;j<cols;j++){float z=0;for(int i=0;i<rows;i++)z+=x[i]*w[AT(i,j,cols)];y[j]=z;}
}
static void addouter(float *g,const float *x,const float *dy,int rows,int cols){
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)g[AT(i,j,cols)]+=x[i]*dy[j];
}
static void backvec(const float *w,const float *dy,float *dx,int rows,int cols){
 for(int i=0;i<rows;i++)for(int j=0;j<cols;j++)dx[i]+=w[AT(i,j,cols)]*dy[j];
}
static float forward(const unsigned char *tokens,int n,int target,Cache *c){
 memset(c,0,sizeof(*c));
 for(int t=0;t<n;t++){
  for(int i=0;i<D;i++)c->x[t][i]=model->emb[tokens[t]][i]+model->pos[t][i];
  matvec(c->x[t],&model->q[0][0],c->q[t],D,D);
  matvec(c->x[t],&model->k[0][0],c->k[t],D,D);
  matvec(c->x[t],&model->v[0][0],c->v[t],D,D);
 }
 for(int t=0;t<n;t++){
  float mx=-1e30f,sum=0;
  for(int j=0;j<=t;j++){
   float z=0;for(int i=0;i<D;i++)z+=c->q[t][i]*c->k[j][i];
   z*=1.0f/sqrtf((float)D);c->att[t][j]=z;if(z>mx)mx=z;
  }
  for(int j=0;j<=t;j++){c->att[t][j]=expf(c->att[t][j]-mx);sum+=c->att[t][j];}
  for(int j=0;j<=t;j++){
   c->att[t][j]/=sum;
   for(int i=0;i<D;i++)c->mix[t][i]+=c->att[t][j]*c->v[j][i];
  }
  float projected[D];
  matvec(c->mix[t],&model->o[0][0],projected,D,D);
  for(int i=0;i<D;i++)c->h[t][i]=c->x[t][i]+projected[i];
  float pre[F];matvec(c->h[t],&model->f1[0][0],pre,D,F);
  for(int j=0;j<F;j++)c->ff[t][j]=tanhf(pre[j]+model->b1[j]);
  matvec(c->ff[t],&model->f2[0][0],c->y[t],F,D);
  for(int i=0;i<D;i++)c->y[t][i]+=c->h[t][i]+model->b2[i];
 }
 float loss=0.f;
 int first=target<0?n-1:0;
 for(int t=first;t<n;t++){
  float peak=-1e30f,sum=0.f;
  for(int b=0;b<V;b++){
   float z=0.f;
   for(int i=0;i<D;i++)z+=c->y[t][i]*model->emb[b][i];
   c->prob[t][b]=z;
   if(z>peak)peak=z;
  }
  for(int b=0;b<V;b++){
   c->prob[t][b]=expf(c->prob[t][b]-peak);
   sum+=c->prob[t][b];
  }
  for(int b=0;b<V;b++)c->prob[t][b]/=sum;
  if(target>=0 && t>=supervision_start)loss-=logf(fmaxf(c->prob[t][tokens[t+1]],1e-20f));
 }
 return target<0?0.f:loss/(float)(n-supervision_start);
}
static void backward(const unsigned char *tokens,int n,Cache *c,Model *g){
 float dy[T][D]={0},dh[T][D]={0},dx[T][D]={0};
 float dq[T][D]={0},dk[T][D]={0},dv[T][D]={0},dmix[T][D]={0};
 for(int t=supervision_start;t<n;t++){
  for(int b=0;b<V;b++){
   float e=(c->prob[t][b]-(b==tokens[t+1]?1.f:0.f))/(float)(n-supervision_start);
   for(int i=0;i<D;i++){
    g->emb[b][i]+=e*c->y[t][i];
    dy[t][i]+=e*model->emb[b][i];
   }
  }
 }
 for(int t=n-1;t>=0;t--){
  float dff[F]={0},dpre[F]={0},da[T]={0};
  addouter(&g->f2[0][0],c->ff[t],dy[t],F,D);
  backvec(&model->f2[0][0],dy[t],dff,F,D);
  for(int i=0;i<D;i++){dh[t][i]+=dy[t][i];g->b2[i]+=dy[t][i];}
  for(int j=0;j<F;j++){dpre[j]=dff[j]*(1-c->ff[t][j]*c->ff[t][j]);g->b1[j]+=dpre[j];}
  addouter(&g->f1[0][0],c->h[t],dpre,D,F);
  backvec(&model->f1[0][0],dpre,dh[t],D,F);
  for(int i=0;i<D;i++)dx[t][i]+=dh[t][i];
  addouter(&g->o[0][0],c->mix[t],dh[t],D,D);
  backvec(&model->o[0][0],dh[t],dmix[t],D,D);
  for(int j=0;j<=t;j++){
   for(int i=0;i<D;i++){da[j]+=dmix[t][i]*c->v[j][i];dv[j][i]+=c->att[t][j]*dmix[t][i];}
  }
  float mean=0;for(int j=0;j<=t;j++)mean+=da[j]*c->att[t][j];
  for(int j=0;j<=t;j++){
   float ds=c->att[t][j]*(da[j]-mean)/sqrtf((float)D);
   for(int i=0;i<D;i++){dq[t][i]+=ds*c->k[j][i];dk[j][i]+=ds*c->q[t][i];}
  }
 }
 for(int t=0;t<n;t++){
  addouter(&g->q[0][0],c->x[t],dq[t],D,D);
  addouter(&g->k[0][0],c->x[t],dk[t],D,D);
  addouter(&g->v[0][0],c->x[t],dv[t],D,D);
  backvec(&model->q[0][0],dq[t],dx[t],D,D);
  backvec(&model->k[0][0],dk[t],dx[t],D,D);
  backvec(&model->v[0][0],dv[t],dx[t],D,D);
  for(int i=0;i<D;i++){g->emb[tokens[t]][i]+=dx[t][i];g->pos[t][i]+=dx[t][i];}
 }
}
typedef struct {uint32_t magic,version,dimensions,context,bytes;uint64_t steps;} Header;
static int save(const char *name,uint64_t steps){
 FILE *check=fopen(name,"rb");if(check){fclose(check);fputs("refusing to overwrite existing checkpoint\n",stderr);return 0;}
 FILE *f=fopen(name,"wb");if(!f){perror(name);return 0;}
 Header h={MAGIC,VERSION,D,T,(uint32_t)sizeof(*model),steps};
 int ok=fwrite(&h,sizeof(h),1,f)==1&&fwrite(model,sizeof(*model),1,f)==1;
 if(fclose(f))ok=0;
 if(!ok)remove(name);
 return ok;
}
static int restore(const char *name){
 FILE *f=fopen(name,"rb");if(!f){perror(name);return 0;}
 Header h;int ok=fread(&h,sizeof(h),1,f)==1&&h.magic==MAGIC&&h.version==VERSION&&h.dimensions==D&&h.context==T&&h.bytes==sizeof(*model);
 if(ok)ok=fread(model,sizeof(*model),1,f)==1&&fgetc(f)==EOF;
 fclose(f);
 if(!ok)fputs("checkpoint invalid/incompatible\n",stderr);
 return ok;
}
static int train(const char *corpus,const char *output,int steps){
 if(steps<1||steps>200000){fputs("steps must be 1..200000\n",stderr);return 2;}
 if(!load_data(corpus))return 1;
 if(data_len<=T+1024){fputs("corpus too short for train/validation split\n",stderr);free(data);return 1;}
 /* Collect boundaries before splitting so held-out examples do not leak
  * through a byte-position cut. Blank lines delimit approved records. */
 size_t *starts=malloc((data_len+1)*sizeof(size_t));
 if(!starts){free(data);return 1;}
 size_t count=0;
 starts[count++]=0;
 for(size_t i=1;i+1<data_len;i++)
  if(data[i-1]=='\n' && data[i]=='\n' && i+1<data_len && data[i+1]!='\n')
   starts[count++]=i+1;
 if(count<2){free(starts);free(data);fputs("corpus needs at least two records\n",stderr);return 1;}
 size_t training_count=count*9/10;
 if(training_count<1)training_count=1;
 init();
 grad=calloc(1,sizeof(*model));moment=calloc(1,sizeof(*model));variance=calloc(1,sizeof(*model));
 if(!grad||!moment||!variance)return 1;
 Cache *c=malloc(sizeof(*c));if(!c)return 1;
 for(int s=1;s<=steps;s++){
  /* Every position now contributes a next-byte target. The old trainer
   * supervised only four bytes per optimizer step, far too sparse for
   * the available approved corpus. */
  memset(grad,0,sizeof(*model));
  float total=0.f;
  for(int sample=0;sample<2;sample++){
   size_t index=(size_t)(rng()%training_count);
   size_t begin=starts[index];
   size_t end=index+1<count?starts[index+1]:data_len;
   while(end>begin && (data[end-1]=='\n'||data[end-1]=='\r'))end--;
   if(end<=begin+1){sample--;continue;}
   size_t answer=begin;
   const char label[]="Answer:";
   for(size_t p=begin;p+sizeof(label)-1<=end;p++)
    if(!memcmp(data+p,label,sizeof(label)-1)){answer=p+sizeof(label)-1;break;}
   /* Crop from the record, keeping as much question context as fits
    * before the answer. Train only answer bytes for labeled records. */
   size_t window=begin;
   if(end-begin>T+1){
    if(answer>begin && end-answer<T)window=end-(T+1);
    else if(answer>begin)window=answer>T/2?answer-T/2:begin;
    else window=begin+(size_t)(rng()%(end-begin-T));
   }
   if(window<begin)window=begin;
   if(window+1>=end)window=end-2;
   int n=(int)((end-window-1)<T?(end-window-1):T);
   if(n<1){sample--;continue;}
   supervision_start=0;
   if(answer>window && answer<window+(size_t)n)
    supervision_start=(int)(answer-window-1);
   total+=forward(data+window,n,0,c);
   backward(data+window,n,c,(Model*)grad);
  }
  supervision_start=0;
  float *w=(float*)model;
  float b1=1.f-powf(0.9f,(float)s),b2=1.f-powf(0.999f,(float)s);
  for(size_t i=0;i<sizeof(*model)/sizeof(float);i++){
   float d=fmaxf(-1.f,fminf(1.f,grad[i]/2.f));
   moment[i]=0.9f*moment[i]+0.1f*d;
   variance[i]=0.999f*variance[i]+0.001f*d*d;
   w[i]-=0.0007f*(moment[i]/b1)/(sqrtf(variance[i]/b2)+1e-8f);
  }
  if(s==1||s%100==0||s==steps){
   float validation=0;
   for(int j=0;j<8;j++){
    size_t index=training_count+(size_t)(rng()%(count-training_count));
    size_t begin=starts[index],end=index+1<count?starts[index+1]:data_len;
    while(end>begin && (data[end-1]=='\n'||data[end-1]=='\r'))end--;
    if(end<=begin+1)continue;
    size_t window=begin;
    if(end-begin>T+1)window=end-T-1;
    int n=(int)((end-window-1)<T?(end-window-1):T);
    supervision_start=0;
    const char label[]="Answer:";
    for(size_t p=window;p+sizeof(label)-1<=end;p++)
     if(!memcmp(data+p,label,sizeof(label)-1)){
      size_t offset=p+sizeof(label)-1-window;
      if(offset<(size_t)n)supervision_start=(int)(offset?offset-1:0);
      break;
     }
    validation+=forward(data+window,n,0,c)/8.f;
   }
   supervision_start=0;
   printf("step=%d train_loss=%.5f validation_loss=%.5f\n",s,total/2.f,validation);
   fflush(stdout);
  }
 }
 int ok=save(output,(uint64_t)steps);
 free(starts);free(c);free(data);free(grad);free(moment);free(variance);free(model);
 return ok?0:1;
}
/* Compatibility only: legacy filtered sampler used by the existing service.
 * Qualification uses explicit unfiltered greedy/sample modes below. */
/* Constrain decoding to corpus-like printable English bytes.
 * This prevents arbitrary high-byte output, but does not claim semantic fluency.
 * A low-temperature, top-k decoder reduces unstable random characters. */
static int generate_legacy(const char *checkpoint,const char *prompt,int length,int requested_seed){
 if(length<1||length>256)return 2;
 init();if(!restore(checkpoint)){free(model);return 1;}
 seed=(uint64_t)(unsigned)requested_seed+1;
 size_t plen=strlen(prompt);if(plen==0||plen>4096){free(model);return 2;}
 size_t capacity=plen+(size_t)length;
 unsigned char *out=malloc(capacity);Cache *c=malloc(sizeof(*c));
 if(!out||!c){free(out);free(c);free(model);return 1;}
 memcpy(out,prompt,plen);size_t used=plen;
 int at_word=(plen>0&&prompt[plen-1]!=' ');
 for(int n=0;n<length;n++){
  int ctx=(int)(used<T?used:T);
  forward(out+used-ctx,ctx,-1,c);
  float weights[V]={0};
  float peak=0;
  for(int b=0;b<V;b++){
   /* ASCII letters, space, and conservative sentence punctuation only. */
   int allowed=(b>='a'&&b<='z')||(b>='A'&&b<='Z')||
               b==' '||b=='.'||b==','||b==';'||b=='!'||b=='?'||
               b=='-'||b=='\''|| (b>='0'&&b<='9');
   if(!allowed)continue;
   if(!at_word&&(b=='.'||b==','||b==';'||b=='!'||b=='?'||b=='-'||b==' '))continue;
   if(at_word&&b==' '&&out[used-1]==' ')continue;
   if(c->prob[ctx-1][b]>peak)peak=c->prob[ctx-1][b];
  }
  double sum=0;
  for(int b=0;b<V;b++){
   int allowed=(b>='a'&&b<='z')||(b>='A'&&b<='Z')||
               b==' '||b=='.'||b==','||b==';'||b=='!'||b=='?'||
               b=='-'||b=='\''||(b>='0'&&b<='9');
   if(!allowed||(!at_word&&(b=='.'||b==','||b==';'||b=='!'||b=='?'||b=='-'||b==' ')))continue;
   if(at_word&&b==' '&&out[used-1]==' ')continue;
   /* Avoid injecting letters with extremely weak model support. */
   if(c->prob[ctx-1][b]<peak*0.10f)continue;
   weights[b]=powf(c->prob[ctx-1][b]/peak,2.5f);
   sum+=weights[b];
  }
  if(!(sum>0))break;
  double pick=((rng()>>40)/16777216.0)*sum;
  int choice=' ';
  for(int b=0;b<V;b++)if(weights[b]>0){pick-=weights[b];if(pick<=0){choice=b;break;}}
  out[used++]=(unsigned char)choice;
  if(choice==' ')at_word=0;
  else if(choice=='.'||choice=='!'||choice=='?'){break;}
  else at_word=1;
 }
 fwrite(out+plen,1,used-plen,stdout);
 putchar('\n');
 free(out);free(c);free(model);return 0;
}
/* Decode the actual 256-byte model distribution. Greedy mode is intended
 * for reproducible qualification; sampling retains temperature 0.4.
 * No character whitelist, word-state rules, or punctuation substitutions. */
static int generate(const char *checkpoint,const char *prompt,int length,int requested_seed,int greedy){
 if(length<1||length>256)return 2;
 init();if(!restore(checkpoint)){free(model);return 1;}
 seed=(uint64_t)(unsigned)requested_seed+1;
 size_t plen=strlen(prompt);if(plen==0||plen>4096){free(model);return 2;}
 size_t capacity=plen+(size_t)length;
 unsigned char *out=malloc(capacity);Cache *c=malloc(sizeof(*c));
 if(!out||!c){free(out);free(c);free(model);return 1;}
 memcpy(out,prompt,plen);size_t used=plen;
 for(int n=0;n<length;n++){
  int ctx=(int)(used<T?used:T);
  forward(out+used-ctx,ctx,-1,c);
  int choice=0;
  for(int b=1;b<V;b++)if(c->prob[ctx-1][b]>c->prob[ctx-1][choice])choice=b;
  float peak=c->prob[ctx-1][choice];
  if(!isfinite(peak)||!(peak>0)){free(out);free(c);free(model);fputs("invalid model probabilities\n",stderr);return 1;}
  if(!greedy){
   double weights[V],sum=0;
   for(int b=0;b<V;b++){weights[b]=pow((double)c->prob[ctx-1][b]/peak,2.5);sum+=weights[b];}
   double pick=((rng()>>40)/16777216.0)*sum;
   for(int b=0;b<V;b++){pick-=weights[b];if(pick<0){choice=b;break;}}
  }
  out[used++]=(unsigned char)choice;
 }
 int ok=fwrite(out+plen,1,used-plen,stdout)==used-plen && putchar('\n')!=EOF;
 free(out);free(c);free(model);return ok?0:1;
}
/* Read-only training-input audit. Mirrors the present record/window rules
 * without running optimizer steps or creating checkpoints. */
static void audit_corpus(void){
 size_t records=0,qa=0,prose=0,short_records=0,clipped=0;
 size_t prose_bytes=0,qa_bytes=0,largest=0;
 for(size_t begin=0;begin<data_len;){
  size_t end=begin;
  while(end+1<data_len && !(data[end]=='\n' && data[end+1]=='\n'))end++;
  if(end+1>=data_len)end=data_len;
  size_t actual=end;
  while(actual>begin && (data[actual-1]=='\n'||data[actual-1]=='\r'))actual--;
  if(actual>begin){
   size_t length=actual-begin;
   if(length>largest)largest=length;
   records++;
   if(length<2)short_records++;
   int labeled=0;
   for(size_t i=begin;i+7<=actual;i++){
    if(memcmp(data+i,"Answer:",7)==0){labeled=1;break;}
   }
   if(labeled){qa++;qa_bytes+=length;}
   else {prose++;prose_bytes+=length;}
   if(length>T+1)clipped++;
  }
  if(end==data_len)break;
  begin=end+2;
  while(begin<data_len && (data[begin]=='\n'||data[begin]=='\r'))begin++;
 }
 printf("audit records=%zu qa=%zu prose=%zu short=%zu\n",
        records,qa,prose,short_records);
 printf("audit qa_bytes=%zu prose_bytes=%zu largest_record=%zu context=%d\n",
        qa_bytes,prose_bytes,largest,T);
 printf("audit long_records=%zu current_training_window_limit=%d\n",
        clipped,T+1);
 if(clipped)
  puts("audit WARNING: existing trainer samples only one initial/context-selected window per record per update; full-length prose coverage is not established.");
 if(prose)
  puts("audit NOTE: unlabeled prose uses next-byte targets, not question-answer supervision.");
}
int main(int argc,char **argv){
 if(argc==3&&!strcmp(argv[1],"inspect")){
  if(!load_data(argv[2]))return 1;
  audit_corpus();
  free(data);data=NULL;return 0;
 }
 if(argc==5&&!strcmp(argv[1],"train"))return train(argv[2],argv[3],atoi(argv[4]));
 if(argc==6&&!strcmp(argv[1],"generate"))return generate_legacy(argv[2],argv[3],atoi(argv[4]),atoi(argv[5]));
 if((argc==7)&&!strcmp(argv[1],"generate")){
  if(argc==7&&strcmp(argv[6],"greedy")&&strcmp(argv[6],"sample"))return 2;
  return generate(argv[2],argv[3],atoi(argv[4]),atoi(argv[5]),argc==7&&!strcmp(argv[6],"greedy"));
 }
 fputs("usage:\n  micro_llm inspect CORPUS_FILE_OR_DIRECTORY\n  micro_llm train CORPUS_FILE_OR_DIRECTORY CHECKPOINT STEPS\n  micro_llm generate CHECKPOINT PROMPT BYTES SEED [sample|greedy]\n",stderr);
 return 2;
}
