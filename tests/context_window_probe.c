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

/* Compare equal-length content mutation with a one-byte position shift.
 * Both retain the exact original target byte. No model parameters change.
 * Padding uses a real leading space and changes both length and attention,
 * so this diagnoses shift sensitivity rather than attributing causality to
 * positional embeddings alone. */
static int sensitivity_probe(const size_t *starts,size_t count,Cache *c){
 size_t train_count=count*9/10;
 if(train_count<1)train_count=1;
 for(int part=0;part<2;part++){
  size_t lo=part?train_count:0,hi=part?count:train_count;
  double base=0,content=0,shift=0;
  size_t used=0,rank_base=0,rank_content=0,rank_shift=0;
  /* Replicable partition-stratified samples from real records.
   * Select a target where a one-byte prefix remains below context T.
   * Preserve target and suffix; replace the first third of context with
   * byte 'x' (equal length), versus prefixing one literal space. */
  for(size_t j=0;j<256;j++){
   size_t index=lo+(j*(hi-lo))/256;
   size_t begin=starts[index],end=index+1<count?starts[index+1]:data_len;
   while(end>begin&&(data[end-1]=='\n'||data[end-1]=='\r'||data[end-1]==0x1e))end--;
   if(end<=begin+16)continue;
   size_t span=end-begin-1;
   size_t offset=16+(j*17u)%(span-15);
   if(offset>=T)offset=T-1;
   size_t k=begin+offset;
   if(k>=end)continue;
   unsigned char original[T+1],changed[T+1],padded[T+1];
   size_t n=k-begin;
   if(n<16||n>=T)continue;
   memcpy(original,data+begin,n);
   memcpy(changed,original,n);
   /* Content intervention does not alter byte count, target, or
    * the final two thirds of the prediction context. */
   for(size_t q=0;q<n/3;q++)changed[q]='x';
   padded[0]=' ';memcpy(padded+1,original,n);
   const unsigned char target=data[k];
   const unsigned char *inputs[3]={original,changed,padded};
   double losses[3];int ranks[3];
   for(int a=0;a<3;a++){
    size_t len=n+(a==2);
    forward(inputs[a],(int)len,-1,c);
    float prob=c->prob[len-1][target];
    losses[a]=-log(fmax((double)prob,1e-30));
    int rank=1;
    for(int b=0;b<V;b++)if(c->prob[len-1][b]>prob)rank++;
    ranks[a]=rank;
   }
   base+=losses[0];content+=losses[1];shift+=losses[2];
   rank_base+=(size_t)ranks[0];rank_content+=(size_t)ranks[1];rank_shift+=(size_t)ranks[2];
   used++;
  }
  if(!used){fprintf(stderr,"no eligible sensitivity samples for %s\n",part?"heldout":"training");return 1;}
  printf("sensitivity partition=%s measured=%zu baseline_nll=%.7f content_nll=%.7f shift_nll=%.7f baseline_rank=%.3f content_rank=%.3f shift_rank=%.3f\n",
   part?"heldout":"training",used,base/used,content/used,shift/used,
   (double)rank_base/used,(double)rank_content/used,(double)rank_shift/used);
 }
 return 0;
}


/* Corpus-only byte baselines. Counts come from the training partition;
 * no held-out target contributes to their fitting. Laplace smoothing
 * avoids zero probabilities and is reported as part of the method. */
static int frequency_probe(const size_t *starts,size_t count,Cache *c){
 size_t train_count=count*9/10;
 if(train_count<1)train_count=1;
 uint64_t uni[V]={0},bi[V][V]={{0}},prev_total[V]={0},total=0;
 for(size_t i=0;i<train_count;i++){
  size_t begin=starts[i],end=i+1<count?starts[i+1]:data_len;
  while(end>begin&&(data[end-1]=='\n'||data[end-1]=='\r'||data[end-1]==0x1e))end--;
  for(size_t k=begin+1;k<end;k++){
   unsigned char prev=data[k-1],target=data[k];
   uni[target]++;bi[prev][target]++;prev_total[prev]++;total++;
  }
 }
 if(!total){fputs("no training bytes for frequency baseline\n",stderr);return 1;}
 for(int part=0;part<2;part++){
  size_t lo=part?train_count:0,hi=part?count:train_count;
  size_t measured=0,model_correct=0,uni_correct=0,bi_correct=0;
  double model_nll=0,uni_nll=0,bi_nll=0;
  for(size_t j=0;j<256;j++){
   size_t index=lo+(j*(hi-lo))/256;
   size_t begin=starts[index],end=index+1<count?starts[index+1]:data_len;
   while(end>begin&&(data[end-1]=='\n'||data[end-1]=='\r'||data[end-1]==0x1e))end--;
   if(end<=begin+1)continue;
   size_t first=begin+1;
   for(size_t p=begin;p+7<=end;p++)if(!memcmp(data+p,"Answer:",7)){first=p+7;break;}
   if(first>=end)continue;
   size_t k=first+((j*17u)%(end-first));
   size_t window=k-begin>(size_t)T?k-T:begin;
   int n=(int)(k-window);
   if(n<1)continue;
   forward(data+window,n,-1,c);
   int top=0,best_uni=0,best_bi=0;
   unsigned char target=data[k],prev=data[k-1];
   for(int b=1;b<V;b++){
    if(c->prob[n-1][b]>c->prob[n-1][top])top=b;
    if(uni[b]>uni[best_uni])best_uni=b;
    if(bi[prev][b]>bi[prev][best_bi])best_bi=b;
   }
   double pu=((double)uni[target]+1)/((double)total+V);
   double pb=((double)bi[prev][target]+1)/((double)prev_total[prev]+V);
   model_nll-=log(fmax((double)c->prob[n-1][target],1e-30));
   uni_nll-=log(pu);bi_nll-=log(pb);
   model_correct+=(top==target);uni_correct+=(best_uni==target);bi_correct+=(best_bi==target);
   measured++;
  }
  if(!measured){fputs("no targets measured\n",stderr);return 1;}
  printf("frequency partition=%s measured=%zu model_nll=%.7f unigram_nll=%.7f bigram_nll=%.7f model_accuracy=%.7f unigram_accuracy=%.7f bigram_accuracy=%.7f\n",
   part?"heldout":"training",measured,model_nll/measured,uni_nll/measured,bi_nll/measured,
   (double)model_correct/measured,(double)uni_correct/measured,(double)bi_correct/measured);
 }
 return 0;
}

int main(int argc,char **argv){
 if(argc!=3 && !(argc==4 && (!strcmp(argv[3],"sensitivity") || !strcmp(argv[3],"frequency")))){fputs("usage: context_window_probe EXPORTED_CORPUS CHECKPOINT [sensitivity|frequency]\n",stderr);return 2;}
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
 if(argc==4){int rc=!strcmp(argv[3],"frequency")?frequency_probe(starts,count,c):sensitivity_probe(starts,count,c);free(c);free(model);free(starts);free(data);return rc;}
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
