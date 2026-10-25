/* Exact teacher-forcing/generation context equivalence, including all targets. */
#define main micro_llm_main
#include "../micro_llm.c"
#undef main
static int exercise(unsigned char *text,size_t len,size_t answer){
 TrainingRecord r={0,len,answer,0};TrainingCorpus corpus={text,&r,1};
 Cache *c=malloc(sizeof(*c));Model *actual=calloc(1,sizeof(*actual)),*expected=calloc(1,sizeof(*expected)),*scratch=malloc(sizeof(*scratch));
 if(!c||!actual||!expected||!scratch)return 1;
 float result=record_pass(&corpus,&r,c,actual,scratch);double reference=0;size_t first=answer?answer:1,targets=len-first;int fail=0;
 for(size_t k=first;k<len;k++){
  int n=k<T?(int)k:T;supervision_start=n-1;
  reference+=forward(text+k-n,n,0,c);
  memset(scratch,0,sizeof(*scratch));backward(text+k-n,n,c,scratch);
  for(size_t j=0;j<sizeof(*expected)/sizeof(float);j++)((float*)expected)[j]+=((float*)scratch)[j]/(float)targets;
 }
 float max=0;
 for(size_t j=0;j<sizeof(*expected)/sizeof(float);j++){
  float delta=fabsf(((float*)actual)[j]-((float*)expected)[j]);if(delta>max)max=delta;
 }
 if(fabs(result-reference/targets)>1e-5||max>1e-5)fail++;
 printf("prefix-equivalence length=%zu answer=%zu targets=%zu loss_error=%g gradient_error=%g %s\n",len,answer,targets,fabs(result-reference/targets),max,fail?"FAIL":"PASS");
 supervision_start=0;free(c);free(actual);free(expected);free(scratch);return fail;
}
int main(void){
 init();int fail=0;unsigned char text[260];
 const char *short_pair="Question: What is an engineer?\nAnswer: An engineer designs systems.";
 memcpy(text,short_pair,strlen(short_pair));fail+=exercise(text,strlen(short_pair),(size_t)(strstr(short_pair,"Answer:")+7-short_pair));
 for(size_t i=0;i<sizeof(text);i++)text[i]=(unsigned char)('a'+i%26);
 fail+=exercise(text,97,96); /* first answer target exactly at window boundary */
 fail+=exercise(text,98,97); /* question exceeds context by one byte */
 fail+=exercise(text,220,37); /* all late answer bytes, not just first window */
 fail+=exercise(text,220,140); /* unavoidable question truncation only */
 fail+=exercise(text,120,0); /* ordinary language records */
  TrainingCorpus a={0},b={0},duplicates={0},labels={0};
 if(!corpus_read("tests/fixtures/alignment/train.txt",&a)||!corpus_read("tests/fixtures/alignment/heldout.txt",&b))return 1;
 if(a.count!=8||b.count!=4||corpora_overlap(&a,&b))fail++;
 if(!corpus_read("tests/fixtures/alignment/train.txt",&duplicates)||!corpora_overlap(&a,&duplicates))fail++;
 for(size_t i=0;i<a.count;i++)if(a.bytes[a.records[i].answer]!=' '||a.records[i].answer<=a.records[i].begin)fail++;
 corpus_free(&a);corpus_free(&b);corpus_free(&duplicates);
 if(!corpus_read("tests/fixtures/alignment/parser.txt",&labels)||labels.count!=2)fail++;
 for(size_t i=0;i<labels.count;i++){
  const TrainingRecord *r=labels.records+i;
  size_t n=r->end-r->answer;
  if(!((n==9&&!memcmp(labels.bytes+r->answer," A label.",9))||(n==5&&!memcmp(labels.bytes+r->answer," Yes.",5))))fail++;
 }
 corpus_free(&labels);
 printf("record independence/first-answer-byte failures=%d\n",fail);

 free(model);printf("alignment failures=%d\n",fail);return fail!=0;
}
