/* STN-Labz checkpoint-only training/inference context comparison.
 * Build: cc -std=c11 -O2 -Wall -Wextra -pedantic tests/context_window_probe.c -lm -o /tmp/context_window_probe
 * Run:   /tmp/context_window_probe EXPORTED_CORPUS CHECKPOINT
 * Does not train, change model weights, change the checkpoint, or generate files.
 * Each sampled target is scored twice with identical preceding source bytes:
 * (1) its exact window start under the current training schedule and
 * (2) the final up-to-T preceding bytes used by inference.
 */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main

static void score(const unsigned char *bytes,size_t k,size_t begin,
                  Cache *c,double *loss,int *rank,int *predicted){
 int n=(int)(k-begin);
 if(n<1||n>T){fprintf(stderr,"invalid context length\n");exit(2);}
 forward(bytes+begin,n,-1,c);
 unsigned char target=bytes[k];
 float p=c->prob[n-1][target];
 *loss=-log(fmax((double)p,1e-30));
 *rank=1;
 *predicted=0;
 float best=-1;
 for(int b=0;b<V;b++){
  float q=c->prob[n-1][b];
  if(q>p)*rank+=1;
  if(q>best){best=q;*predicted=b;}
 }
}
int main(int argc,char **argv){
 if(argc!=3){fputs("usage: context_window_probe EXPORTED_CORPUS CHECKPOINT\n",stderr);return 2;}
 if(!load_data(argv[1]))return 1;
 /* Same record delimiter precedence as train() and evaluate_corpus(). */
 size_t *starts=malloc((data_len+1)*sizeof(*starts));
 if(!starts){free(data);return 1;}
 size_t count=0;starts[count++]=0;
 if(memchr(data,0x1e,data_len)){
  for(size_t i=0;i+1<data_len;i++)if(data[i]==0x1e)starts[count++]=i+1;
  if(starts[count-1]==data_len)count--;
 }else{
  for(size_t i=1;i+1<data_len;i++)
   if(data[i-1]=='\n'&&data[i]=='\n'&&data[i+1]!='\n')starts[count++]=i+1;
 }
 if(count<2){fputs("at least two records required\n",stderr);free(starts);free(data);return 1;}
 init();
 if(!restore(argv[2])){free(model);free(starts);free(data);return 1;}
 Cache *c=malloc(sizeof(*c));
 if(!c){free(model);free(starts);free(data);return 1;}
 size_t training_count=count*9/10;
 if(training_count<1)training_count=1;
 puts("partition,record,target_offset,training_context,inference_context,training_nll,inference_nll,training_rank,inference_rank,training_best,inference_best,target");
 for(int part=0;part<2;part++){
  size_t lo=part?training_count:0,hi=part?count:training_count;
  size_t measured=0,different=0;
  double tr_loss=0,inf_loss=0;
  /* At most 256 deterministic records in each partition, exactly as in
   * checkpoint evaluation; skip invalid records rather than forge samples. */
  const size_t budget=256;
  for(size_t j=0;j<budget;j++){
   size_t index=lo+(j*(hi-lo))/budget;
   size_t begin=starts[index],end=index+1<count?starts[index+1]:data_len;
   while(end>begin&&(data[end-1]=='\n'||data[end-1]=='\r'||data[end-1]==0x1e))end--;
   if(end<=begin+1)continue;
   size_t first=begin+1;
   for(size_t p=begin;p+7<=end;p++)if(!memcmp(data+p,"Answer:",7)){first=p+7;break;}
   if(first>=end)continue;
   size_t k=first+((j*17u)%(end-first));
   /* The current training loop retains about half a context for each
    * chunk. Find the actual training window covering this target. */
   size_t next=first,train_begin=begin;int found=0;
   while(next<end){
    size_t window=next>begin+(size_t)(T/2)?next-(size_t)(T/2)-1:begin;
    size_t available=end-window-1;
    int n=(int)(available<(size_t)T?available:(size_t)T);
    if(n<1)break;
    size_t supervised=(size_t)n-(next-window-1);
    if(k>=next&&k<next+supervised){train_begin=window;found=1;break;}
    next+=supervised;
   }
   if(!found){fprintf(stderr,"could not locate training window\n");free(c);free(model);free(starts);free(data);return 1;}
   size_t infer_begin=k-begin>(size_t)T?k-T:begin;
   double tl,il;int tr,ir,tb,ib;
   score(data,k,train_begin,c,&tl,&tr,&tb);
   score(data,k,infer_begin,c,&il,&ir,&ib);
   printf("%s,%zu,%zu,%zu,%zu,%.7f,%.7f,%d,%d,%d,%d,%u\n",
          part?"heldout":"training",index,k-begin,k-train_begin,k-infer_begin,
          tl,il,tr,ir,tb,ib,(unsigned)data[k]);
   measured++;different+=(train_begin!=infer_begin);tr_loss+=tl;inf_loss+=il;
  }
  fprintf(stderr,"partition=%s measured=%zu different_context_lengths=%zu mean_training_window_nll=%.7f mean_inference_window_nll=%.7f\n",
          part?"heldout":"training",measured,different,
          measured?tr_loss/measured:0.0,measured?inf_loss/measured:0.0);
  if(!measured){free(c);free(model);free(starts);free(data);return 1;}
 }
 free(c);free(model);free(starts);free(data);
 return 0;
}
