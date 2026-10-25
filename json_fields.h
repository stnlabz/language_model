/* Minimal strict JSON object field reader, no external dependencies.
 * Returns -1 on malformed data. Values are decoded JSON strings, true/false/null.
 * Fields not requested are still syntax-checked; nested values are rejected.
 */
#ifndef STN_JSON_FIELDS_H
#define STN_JSON_FIELDS_H
#include <ctype.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
typedef struct {char *dst;size_t cap;const char *key;int type;int found;} JField;
static void jspace(const char **p){while(**p==' '||**p=='\n'||**p=='\r'||**p=='\t')++*p;}
static int jhex(char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;if(c>='A'&&c<='F')return c-'A'+10;return -1;}
static int jutf(char *out,size_t cap,size_t *n,unsigned cp){
 if(cp<=0x7f){if(*n+1>=cap)return -1;out[(*n)++]=(char)cp;return 0;}
 if(cp<=0x7ff){if(*n+2>=cap)return -1;out[(*n)++]=(char)(0xc0|(cp>>6));out[(*n)++]=(char)(0x80|(cp&63));return 0;}
 if(cp<=0xffff){if(*n+3>=cap)return -1;out[(*n)++]=(char)(0xe0|(cp>>12));out[(*n)++]=(char)(0x80|((cp>>6)&63));out[(*n)++]=(char)(0x80|(cp&63));return 0;}
 if(cp>0x10ffff||*n+4>=cap)return -1;
 out[(*n)++]=(char)(0xf0|(cp>>18));out[(*n)++]=(char)(0x80|((cp>>12)&63));out[(*n)++]=(char)(0x80|((cp>>6)&63));out[(*n)++]=(char)(0x80|(cp&63));return 0;
}
static int jstr(const char **p,char *out,size_t cap){
 if(*(*p)++!='"'||cap==0)return -1;
 size_t n=0;
 while(**p&&**p!='"'){
  unsigned char c=(unsigned char)*(*p)++;
  if(c<0x20)return -1;
  if(c=='\\'){
   c=(unsigned char)*(*p)++;
   switch(c){
    case '"':case '\\':case '/':break;
    case 'n':c='\n';break;case 'r':c='\r';break;case 't':c='\t';break;
    case 'b':c='\b';break;case 'f':c='\f';break;
    case 'u':{
     unsigned cp=0;for(int i=0;i<4;i++){int h=jhex(**p);if(h<0)return -1;cp=(cp<<4)|(unsigned)h;++*p;}
     if(cp>=0xd800&&cp<=0xdbff){
      if((*p)[0]!='\\'||(*p)[1]!='u')return -1;*p+=2;
      unsigned lo=0;for(int i=0;i<4;i++){int h=jhex(**p);if(h<0)return -1;lo=(lo<<4)|(unsigned)h;++*p;}
      if(lo<0xdc00||lo>0xdfff)return -1;
      cp=0x10000+((cp-0xd800)<<10)+(lo-0xdc00);
     }else if(cp>=0xdc00&&cp<=0xdfff)return -1;
     if(jutf(out,cap,&n,cp))return -1;
     continue;
    }
    default:return -1;
   }
  }
  if(n+1>=cap)return -1;out[n++]=(char)c;
 }
 if(**p!='"')return -1;++*p;out[n]=0;return 0;
}
static int jfields(const char *p,JField *f,size_t fields){
 jspace(&p);if(*p++!='{')return -1;jspace(&p);
 if(*p=='}'){p++;jspace(&p);return *p? -1:0;}
 for(;;){
  char key[128],value[65536];int type=0;
  if(jstr(&p,key,sizeof(key)))return -1;jspace(&p);
  if(*p++!=':')return -1;jspace(&p);
  if(*p=='"'){if(jstr(&p,value,sizeof(value)))return -1;type=1;}
  else {
   size_t i=0;
   while(*p&&*p!=','&&*p!='}'&&!isspace((unsigned char)*p)){
    if(i+1>=sizeof(value))return -1;value[i++]=*p++;
   }
   value[i]=0;
   if(!strcmp(value,"true"))type=2;
   else if(!strcmp(value,"false"))type=3;
   else if(!strcmp(value,"null"))type=4;
   else {
    size_t j=0;if(value[j]=='-')j++;
    if(!value[j])return -1;
    for(;value[j];j++)if(value[j]<'0'||value[j]>'9')return -1;
    type=5;
   }
  }
  for(size_t i=0;i<fields;i++)if(!strcmp(key,f[i].key)){
   if(f[i].found||strlen(value)+1>f[i].cap)return -1;
   strcpy(f[i].dst,value);f[i].type=type;f[i].found=1;
  }
  jspace(&p);
  if(*p=='}'){++p;jspace(&p);return *p?-1:0;}
  if(*p++!=',')return -1;jspace(&p);
 }
}
#endif
