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
#define VOC 768
#define MAXTOK 500000
#define MAXWORD 48
#define EPOCHS 60
#define BPTT 24
#define LR 0.003f
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
 int token,target,reset;
 float prev[H],z[H],r[H],n[H],h[H],prob[VOC];
} tick;
static int lookup(const char *p){for(int i=3;i<nw;i++)if(!strcmp(p,words[i].text))return i;return 2;}
static void vocabulary(const char *p){
 int found=-1;
 for(int i=3;i<nw;i++)if(!strcmp(p,words[i].text)){found=i;break;}
 if(found>=0){words[found].freq++;return;}
 if(nw<VOC){strcpy(words[nw].text,p);words[nw++].freq=1;return;}
 /* bounded frequent-word replacement; preserve higher-frequency terms */
 int low=3;for(int i=4;i<nw;i++)if(words[i].freq<words[low].freq)low=i;
 if(words[low].freq==1){strcpy(words[low].text,p);words[low].freq=1;}
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
    buf[len]=0;if(build)vocabulary(buf);
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
 }
}
static double learn(net *m,tick *ts,int count){
 net *g=calloc(1,sizeof(*g));if(!g)return -1;
 float future[H]={0};double loss=0;
 for(int z=count-1;z>=0;z--){
  tick *t=&ts[z];float dh[H],dn[H],dz[H],dr[H],back[H],de[H]={0};
  float err[VOC];for(int k=0;k<nw;k++){err[k]=t->prob[k]-(k==t->target);g->bo[k]+=err[k];}
  loss-=log(fmax((double)t->prob[t->target],1e-25));
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
 if(fclose(f))ok=0;return ok;
}
static int restore(const char *path,net *m){
 FILE *f=fopen(path,"rb");if(!f)return 0;uint32_t magic=0;
 int ok=fread(&magic,4,1,f)==1&&magic==0x57524731&&fread(&nw,sizeof(nw),1,f)==1&&nw>=10&&nw<=VOC&&fread(words,sizeof(words),1,f)==1&&fread(m,sizeof(*m),1,f)==1&&fgetc(f)==EOF;
 fclose(f);return ok;
}
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
 tick t;float h[H]={0};int prev=0,last=-1,words_out=0;
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
   if(k==1&&words_out<6)continue;
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
  memcpy(h,t.h,sizeof(h));prev=choice;
 }
 fputs(".\n",f);
 return fclose(f)==0&&words_out>=6;
}
int main(int argc,char **argv){
 if(argc!=6||strcmp(argv[1],"train")){fputs("usage: word_gru train TRAIN VALID MODEL SAMPLE\n",stderr);return 2;}
 size_t tn=0,vn=0;unsigned char *tr=load(argv[2],&tn),*va=load(argv[3],&vn);
 if(!tr||!va){fputs("training corpus unavailable\n",stderr);free(tr);free(va);return 1;}
 strcpy(words[0].text,"<bos>");strcpy(words[1].text,"<eos>");strcpy(words[2].text,"<unk>");
 int count=0;int *scratch=tokenize(tr,tn,&count,1);free(scratch);
 int nt=0,nv=0;int *train=tokenize(tr,tn,&nt,0),*valid=tokenize(va,vn,&nv,0);
 free(tr);free(va);
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
    ts[n].target=target;ts[n].reset=prev==0;
    memcpy(h,ts[n].h,sizeof(h));prev=target;n++;
    if(target==1){memset(h,0,sizeof(h));prev=0;}
   }
   double l=learn(m,ts,n);if(l<0){free(m);free(best);free(train);free(valid);return 1;}total+=l;
  }
  double vl=evaluate(m,valid,nv);
  printf("epoch=%d train_loss=%.5f valid_loss=%.5f\n",epoch+1,total/nt,vl);
  if(vl<best_loss-0.0005){best_loss=vl;memcpy(best,m,sizeof(*m));bad_epochs=0;}
  else if(++bad_epochs>=7){printf("early_stop_epoch=%d\n",epoch+1);break;}
 }
 build_transitions(train,nt);
 int ok=transitions&&save(argv[4],best)&&produce(best,argv[5]);
 printf("tokens=%d vocabulary=%d best_validation_loss=%.5f\n",nt,nw,best_loss);
 free(m);free(best);free(moment1);free(moment2);free(transitions);free(train);free(valid);return ok?0:1;
}