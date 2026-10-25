/* Frozen-checkpoint teacher-forced versus free-running next-byte diagnosis. */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main
static int top(const float *p){int b=0;for(int i=1;i<V;i++)if(p[i]>p[b])b=i;return b;}
static int rank_byte(const float *p,int b){int rank=1;for(int i=0;i<V;i++)if(p[i]>p[b])rank++;return rank;}
static void distribution(const unsigned char *text,size_t used,Cache *c){int n=used<T?(int)used:T;forward(text+used-n,n,-1,c);}
static float *last(Cache *c,size_t used){return c->prob[used<T?used-1:T-1];}
int main(int argc,char **argv){
 if(argc!=4){fputs("usage: generation_trace CHECKPOINT REFERENCE_TEXT OUTPUT_DIR\n",stderr);return 2;}
 unsigned char reference[8192];FILE *f=fopen(argv[2],"rb");if(!f)return 1;
 size_t len=fread(reference,1,sizeof(reference),f);int too_long=!feof(f);fclose(f);if(too_long)return 1;
 while(len&&(reference[len-1]=='\r'||reference[len-1]=='\n'))len--;
 const size_t prefix=80;if(len<=prefix||len*2>sizeof(reference))return 2;
 f=fopen(argv[1],"rb");Header header;if(!f)return 1;int header_ok=fread(&header,sizeof(header),1,f)==1;fclose(f);if(!header_ok)return 1;
 printf("checkpoint_version=%u checkpoint_steps=%llu\n",header.version,(unsigned long long)header.steps);
 init();if(!restore(argv[1]))return 1;
 Cache *teacher=malloc(sizeof(*teacher)),*self=malloc(sizeof(*self)),*whole=malloc(sizeof(*whole));if(!teacher||!self||!whole)return 1;
 if(len-1>T)return 2;
 forward(reference,(int)len-1,0,whole);
 double causal_difference=0;
 unsigned char generated[8192],forced[8192];memcpy(generated,reference,prefix);memcpy(forced,reference,prefix);
 char path[4096];snprintf(path,sizeof(path),"%s/trace.csv",argv[3]);FILE *csv=fopen(path,"wb");if(!csv)return 1;
 fputs("step,context_bytes,expected_byte,teacher_top,teacher_p_expected,teacher_rank,free_top,free_p_expected,free_rank,distribution_l1\n",csv);
 size_t teacher_hits=0,free_hits=0,first_wrong=0,repeat_start=0,repeat_unit=0;double teacher_loss=0,free_loss=0;
 for(size_t j=0;j<len;j++){
  size_t used=prefix+j;distribution(generated,used,self);float *sp=last(self,used);int choice=top(sp);
  if(j==0)for(int b=0;b<V;b++)if(rank_byte(sp,b)<=7)printf("initial_candidate byte=%d rank=%d probability=%.9g\n",b,rank_byte(sp,b),sp[b]);
  if(used<len){
   distribution(reference,used,teacher);float *tp=last(teacher,used);int expected=reference[used],tt=top(tp);double l1=0;
   for(int b=0;b<V;b++){l1+=fabs((double)tp[b]-sp[b]);double delta=fabs((double)tp[b]-whole->prob[used-1][b]);if(delta>causal_difference)causal_difference=delta;}
   teacher_hits+=tt==expected;free_hits+=choice==expected;
   teacher_loss-=log(fmax(tp[expected],1e-30));free_loss-=log(fmax(sp[expected],1e-30));
   if(choice!=expected&&!first_wrong)first_wrong=j+1;
   fprintf(csv,"%zu,%zu,%d,%d,%.9g,%d,%d,%.9g,%d,%.9g\n",j+1,used<T?used:T,expected,tt,tp[expected],rank_byte(tp,expected),choice,sp[expected],rank_byte(sp,expected),l1);
  }else fprintf(csv,"%zu,%zu,NA,NA,NA,NA,%d,NA,NA,NA\n",j+1,used<T?used:T,choice);
  generated[used]=(unsigned char)choice;
  if(!repeat_start)for(size_t unit=2;unit<=8;unit++)if(j+1>=unit*3){
   size_t start=used+1-unit*3;
   if(!memcmp(generated+start,generated+start+unit,unit)&&!memcmp(generated+start,generated+start+2*unit,unit)){repeat_start=start-prefix+1;repeat_unit=unit;break;}
  }
 }
 fclose(csv);
 /* Diagnostic intervention only: supply the one correct first continuation
  * byte, then allow the unchanged model to generate all later bytes. */
 forced[prefix]=reference[prefix];
 for(size_t j=1;j<len;j++){size_t used=prefix+j;distribution(forced,used,self);forced[used]=(unsigned char)top(last(self,used));}
 snprintf(path,sizeof(path),"%s/free.txt",argv[3]);f=fopen(path,"wb");if(!f)return 1;fwrite(generated+prefix,1,len,f);fputc('\n',f);fclose(f);
 snprintf(path,sizeof(path),"%s/first-byte-corrected.txt",argv[3]);f=fopen(path,"wb");if(!f)return 1;fwrite(forced+prefix,1,len,f);fputc('\n',f);fclose(f);
 printf("first_tripled_fragment_start=%zu fragment_bytes=%zu\n",repeat_start,repeat_unit);
 printf("reference_bytes=%zu prompt_bytes=%zu reference_continuation_bytes=%zu first_wrong_generated_byte=%zu\n",len,prefix,len-prefix,first_wrong);
 printf("prompt=%.*s\nreference_continuation=%.*s\n",(int)prefix,reference,(int)(len-prefix),reference+prefix);
 printf("teacher_accuracy=%zu/%zu teacher_mean_loss=%.6f free_reference_accuracy=%zu/%zu free_reference_mean_loss=%.6f\n",teacher_hits,len-prefix,teacher_loss/(len-prefix),free_hits,len-prefix,free_loss/(len-prefix));
 fputs("free=",stdout);fwrite(generated+prefix,1,len,stdout);fputs("\nfirst_byte_corrected=",stdout);fwrite(forced+prefix,1,len,stdout);putchar('\n');
 printf("training_full_prefix_vs_inference_max_probability_difference=%.9g\n",causal_difference);
 free(whole);free(teacher);free(self);free(model);return causal_difference>1e-6;
}
