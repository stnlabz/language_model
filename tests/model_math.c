/* Native mathematical qualification of the production implementation. */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main
#include <stddef.h>
static int checks(const char *checkpoint,int mask){
 init(); supervision_start=mask; printf("gradient supervision_start=%d\n",mask); if(checkpoint && !restore(checkpoint))return 1; Cache *c=malloc(sizeof(*c)),*other=malloc(sizeof(*other));
 Model *g=calloc(1,sizeof(*g));
 unsigned char tok[]="An engineer designs systems.";
 const int n=20; forward(tok,n,0,c); backward(tok,n,c,g);
 size_t starts[]={offsetof(Model,emb),offsetof(Model,pos),offsetof(Model,q),offsetof(Model,k),offsetof(Model,v),offsetof(Model,o),offsetof(Model,f1),offsetof(Model,f2),offsetof(Model,b1),offsetof(Model,b2),sizeof(Model)};
 const char *names[]={"emb","pos","q","k","v","o","f1","f2","b1","b2"};
 float *w=(float*)model,*gg=(float*)g; int failed=0;
 for(int family=0;family<10;family++){
  size_t a=starts[family]/sizeof(float),b=starts[family+1]/sizeof(float),best=a;
  for(size_t i=a;i<b;i++)if(fabsf(gg[i])>fabsf(gg[best]))best=i;
  float worst=0;
  for(int j=0;j<8;j++){
   size_t i=j==0?best:a+(size_t)(rng()%(b-a));float old=w[i],eps=0.003f;
   w[i]=old+eps;float plus=forward(tok,n,0,other);
   w[i]=old-eps;float minus=forward(tok,n,0,other);w[i]=old;
   float numeric=(plus-minus)/(2*eps),err=fabsf(numeric-gg[i]);if(err>worst)worst=err;
   if(err>0.0004f+0.03f*fabsf(gg[i])){printf("FAIL %s index=%zu analytic=%g numeric=%g\n",names[family],i,gg[i],numeric);failed++;}
  }
  printf("gradient family=%s max_absolute_error=%g\n",names[family],worst);
 }
 forward(tok,n,0,c);tok[15]='X';forward(tok,n,0,other);
 for(int t=0;t<15;t++)for(int b=0;b<V;b++)if(c->prob[t][b]!=other->prob[t][b])failed++;
 for(int t=0;t<n;t++){double sum=0;for(int j=0;j<T;j++){if(j>t&&c->att[t][j]!=0)failed++;sum+=c->att[t][j];}if(fabs(sum-1)>1e-5)failed++;}
 printf("math qualification: %s failures=%d\n",failed?"FAIL":"PASS",failed);
 free(c);free(other);free(g);free(model);supervision_start=0;return failed!=0;
}
static int greedy(const char *path,const char *prompt,int count){
 init();if(!restore(path))return 1;Cache *c=malloc(sizeof(*c));
 unsigned char out[8192];size_t used=strlen(prompt);if(!used||used+count>sizeof(out))return 2;memcpy(out,prompt,used);
 for(int j=0;j<count;j++){int n=used<T?(int)used:T;forward(out+used-n,n,-1,c);int best=0;for(int b=1;b<V;b++)if(c->prob[n-1][b]>c->prob[n-1][best])best=b;out[used++]=(unsigned char)best;putchar(best);}putchar('\n');free(c);free(model);return 0;
}
int main(int argc,char **argv){if(argc==1)return checks(NULL,0)+checks(NULL,7);if(argc==3&&!strcmp(argv[1],"check"))return checks(argv[2],0)+checks(argv[2],7);if(argc==5&&!strcmp(argv[1],"greedy"))return greedy(argv[2],argv[3],atoi(argv[4]));return 2;}
