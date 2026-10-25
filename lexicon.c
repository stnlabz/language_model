/* STN-Labz native lexical inventory from operator-approved text.
 * Independent of the v4 checkpoint; no vocabulary is fabricated.
 * Input: an approved corpus file, or "-" for standard input.
 * Output: lexical entries with corpus frequency, tab-separated.
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#define BUCKETS 131071u
#define MAX_WORD 63
#define MAX_BYTES 50000000u
typedef struct Word {struct Word *next;char text[MAX_WORD+1];uint32_t count;} Word;
static Word *buckets[BUCKETS];
static uint32_t entries=0;
static uint64_t tokens=0;
static unsigned hash_word(const char *s){
 uint32_t h=2166136261u;
 for(;*s;s++){h^=(unsigned char)*s;h*=16777619u;}
 return h%BUCKETS;
}
static int add_word(const char *word){
 unsigned h=hash_word(word);
 for(Word *p=buckets[h];p;p=p->next)
  if(strcmp(p->text,word)==0){
   if(p->count==UINT32_MAX)return 0;
   p->count++;tokens++;return 1;
  }
 Word *p=calloc(1,sizeof(*p));
 if(!p)return 0;
 strcpy(p->text,word);p->count=1;
 p->next=buckets[h];buckets[h]=p;
 entries++;tokens++;return 1;
}
static int cmp_words(const void *a,const void *b){
 const Word *x=*(const Word *const *)a,*y=*(const Word *const *)b;
 return strcmp(x->text,y->text);
}
int main(int argc,char **argv){
 if(argc!=2){fputs("usage: lexicon APPROVED_CORPUS_FILE|\\-\n",stderr);return 2;}
 FILE *f=strcmp(argv[1],"-")==0?stdin:fopen(argv[1],"rb");
 if(!f){perror(argv[1]);return 1;}
 char word[MAX_WORD+1];size_t n=0,bytes=0;
 int ch,overflow=0,ok=1;
 while((ch=fgetc(f))!=EOF){
  if(++bytes>MAX_BYTES){fputs("lexicon: 50MB input limit exceeded\n",stderr);ok=0;break;}
  unsigned char c=(unsigned char)ch;
  if(c>='A'&&c<='Z')c=(unsigned char)(c-'A'+'a');
  if(c>='a'&&c<='z'){
   if(n<MAX_WORD)word[n++]=(char)c;else overflow=1;
  }else{
   if(n&&!overflow){word[n]=0;if(!add_word(word)){ok=0;break;}}
   n=0;overflow=0;
  }
 }
 if(ok&&ferror(f))ok=0;
 if(ok&&n&&!overflow){word[n]=0;if(!add_word(word))ok=0;}
 if(f!=stdin&&fclose(f))ok=0;
 if(!ok){fputs("lexicon: input processing failed\n",stderr);return 1;}
 Word **sorted=malloc((size_t)entries*sizeof(*sorted));
 if(!sorted&&entries)return 1;
 size_t used=0;
 for(unsigned i=0;i<BUCKETS;i++)for(Word *p=buckets[i];p;p=p->next)sorted[used++]=p;
 qsort(sorted,used,sizeof(*sorted),cmp_words);
 fprintf(stderr,"lexicon_entries=%u observed_words=%llu input_bytes=%zu\n",
         entries,(unsigned long long)tokens,bytes);
 for(size_t i=0;i<used;i++)
  if(printf("%s\t%u\n",sorted[i]->text,sorted[i]->count)<0){ok=0;break;}
 free(sorted);
 return ok?0:1;
}
