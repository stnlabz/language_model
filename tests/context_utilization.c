/* Measure whether preceding text affects next-byte predictions.
 * Uses the production forward pass, not a scripted answer or decoder filter.
 * This is a diagnostic: significant sensitivity alone does not prove fluency.
 */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main
int main(int argc,char **argv){
 if(argc!=2){fputs("usage: context_utilization CHECKPOINT\n",stderr);return 2;}
 init();
 if(!restore(argv[1])){free(model);return 1;}
 Cache *a=malloc(sizeof(*a)),*b=malloc(sizeof(*b));
 if(!a||!b){free(a);free(b);free(model);return 1;}
 const char *left="The fresh bread is";
 const char *right="The metal plate is";
 /* Equal byte length and final suffix: neither absolute position nor the
  * immediate suffix alone can explain distribution differences. */
 int na=(int)strlen(left),nb=(int)strlen(right);
 if(na!=nb){fputs("FAIL: context controls differ in length\n",stderr);free(a);free(b);free(model);return 1;}
 forward((const unsigned char*)left,na,-1,a);
 forward((const unsigned char*)right,nb,-1,b);
 double variation=0;
 int best_a=0,best_b=0;
 for(int i=0;i<V;i++){
  variation+=fabs((double)a->prob[na-1][i]-b->prob[nb-1][i]);
  if(a->prob[na-1][i]>a->prob[na-1][best_a])best_a=i;
  if(b->prob[nb-1][i]>b->prob[nb-1][best_b])best_b=i;
 }
 printf("context_sensitivity_l1=%.8f\n",variation);
 printf("context_top_bytes=%d,%d\n",best_a,best_b);
 free(a);free(b);free(model);
 if(!isfinite(variation)){fputs("FAIL: invalid context sensitivity\n",stderr);return 1;}
 return 0;
}
