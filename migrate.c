/* STN-Labz native Windows JSONL -> byte corpus conversion.
 * Accepted schema: input,response,source,approved. Source is read-only.
 * Output is new-file-only. Dialogues remain Q/A; Corpus LESSON entries
 * become statements. Context and Observations remain in the snapshot only.
 */
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <ctype.h>
#define LINE_MAX_BYTES 8192
#define CORPUS_LIMIT (16u*1024u*1024u)
static int ws(unsigned char c) { return c==' ' || c=='\t' || c=='\r' || c=='\n'; }
static void skip(const char **p) { while(ws((unsigned char)**p)) ++*p; }
static int hex(int c) {
    if(c>='0'&&c<='9')return c-'0';
    if(c>='a'&&c<='f')return c-'a'+10;
    if(c>='A'&&c<='F')return c-'A'+10;
    return -1;
}
static int utf8(char *out,size_t cap,size_t *n,unsigned c) {
    unsigned char b[4];int k;
    if(c<=0x7f){b[0]=(unsigned char)c;k=1;}
    else if(c<=0x7ff){b[0]=(unsigned char)(0xc0|(c>>6));b[1]=(unsigned char)(0x80|(c&63));k=2;}
    else if(c<=0xffff){if(c>=0xd800&&c<=0xdfff)return 0;b[0]=(unsigned char)(0xe0|(c>>12));b[1]=(unsigned char)(0x80|((c>>6)&63));b[2]=(unsigned char)(0x80|(c&63));k=3;}
    else if(c<=0x10ffff){b[0]=(unsigned char)(0xf0|(c>>18));b[1]=(unsigned char)(0x80|((c>>12)&63));b[2]=(unsigned char)(0x80|((c>>6)&63));b[3]=(unsigned char)(0x80|(c&63));k=4;}
    else return 0;
    if(*n+(size_t)k>=cap)return 0;
    memcpy(out+*n,b,(size_t)k);*n+=(size_t)k;out[*n]=0;return 1;
}
static int read_u4(const char **p,unsigned *v) {
    unsigned x=0;int h;
    for(int i=0;i<4;i++){h=hex((unsigned char)(*p)[i]);if(h<0)return 0;x=x*16+(unsigned)h;}
    *p+=4;*v=x;return 1;
}
static int json_string(const char **p,char *out,size_t cap) {
    size_t n=0;
    if(*(*p)++!='"')return 0;
    while(**p && **p!='"') {
        unsigned c=(unsigned char)*(*p)++;
        if(c=='\\'){
            c=(unsigned char)*(*p)++;
            if(c=='u'){
                unsigned u=0,v=0;
                if(!read_u4(p,&u))return 0;
                if(u>=0xd800&&u<=0xdbff){
                    if((*p)[0]!='\\'||(*p)[1]!='u')return 0;
                    *p+=2;if(!read_u4(p,&v)||v<0xdc00||v>0xdfff)return 0;
                    u=0x10000+((u-0xd800)<<10)+(v-0xdc00);
                } else if(u>=0xdc00&&u<=0xdfff)return 0;
                if(!utf8(out,cap,&n,u))return 0;continue;
            }
            switch(c){
                case 'n':c=' ';break;case 'r':c=' ';break;case 't':c=' ';break;
                case '"':break;case '\\':break;case '/':break;
                case 'b':c=' ';break;case 'f':c=' ';break;
                default:return 0;
            }
        }
        if(c==0||c<32)return 0;
        if(c=='\n'||c=='\r')c=' ';
        if(n+1>=cap)return 0;
        out[n++]=(char)c;
    }
    if(**p!='"')return 0;
    ++*p;out[n]=0;return 1;
}
static int line_pair(const char *p,char *question,char *answer) {
    char key[64],value[4096];int mask=0;
    skip(&p);if(*p++!='{')return 0;
    for(;;) {
        skip(&p);if(*p=='}'){p++;break;}
        if(!json_string(&p,key,sizeof(key)))return 0;
        skip(&p);if(*p++!=':')return 0;skip(&p);
        if(!strcmp(key,"approved")) {
            if(mask&8||strncmp(p,"true",4))return 0;p+=4;mask|=8;
        } else {
            if(!json_string(&p,value,sizeof(value)))return 0;
            if(!strcmp(key,"input")) {
                if(mask&1||!value[0]||strlen(value)>1024)return 0;
                strcpy(question,value);mask|=1;
            } else if(!strcmp(key,"response")) {
                if(mask&2||!value[0]||strlen(value)>2048)return 0;
                strcpy(answer,value);mask|=2;
            } else if(!strcmp(key,"source")) {
                if(mask&4||strcmp(value,"digit"))return 0;mask|=4;
            } else return 0;
        }
        skip(&p);if(*p==','){p++;continue;}
        if(*p=='}'){p++;break;}return 0;
    }
    skip(&p);
    return mask==15 && !*p;
}
int main(int argc,char **argv) {
    if(argc!=4||strcmp(argv[1],"convert")) {
        fprintf(stderr,"usage: %s convert <snapshot.jsonl> <new-corpus.txt>\n",argv[0]);return 2;
    }
    if(!strcmp(argv[2],argv[3]))return 2;
    FILE *src=fopen(argv[2],"rb");if(!src){perror("snapshot");return 1;}
    FILE *check=fopen(argv[3],"rb");
    if(check){fclose(check);fclose(src);fputs("output exists\n",stderr);return 1;}
    /* Build in memory; never publish a partially converted output. */
    char *corpus=malloc(CORPUS_LIMIT+1),line[LINE_MAX_BYTES+1],q[4096],a[4096];
    if(!corpus){fclose(src);return 1;}
    size_t total=0,rows=0,lessons=0,context=0,observations=0,dialogues=0;
    int result=1;
    while(fgets(line,sizeof(line),src)) {
        size_t length=strlen(line);
        if(!length||line[length-1]!='\n'){fprintf(stderr,"invalid/incomplete line %zu\n",rows+1);goto done;}
        line[length-1]=0;
        if(length>1&&line[length-2]=='\r')line[length-2]=0;
        if(!line_pair(line,q,a)){fprintf(stderr,"invalid record at line %zu\n",rows+1);goto done;}
        int count;
        if(!strncmp(q,"Digit Context record ",21)){
            context++;rows++;continue;
        } else if(!strncmp(q,"Digit Observation record ",25)){
            observations++;rows++;continue;
        } else if(!strncmp(q,"Digit Corpus record ",20)){
            /* Corpus entries are statements, not questions. Keep only
             * explicitly labeled LESSON entries; reject unknown categories. */
            if(!strstr(q,", category LESSON,")){
                fprintf(stderr,"unsupported Corpus category at line %zu\n",rows+1);
                goto done;
            }
            count=snprintf(corpus+total,CORPUS_LIMIT+1-total,"%s\n",a);
            lessons++;
        } else {
            count=snprintf(corpus+total,CORPUS_LIMIT+1-total,"Question: %s Answer: %s\n",q,a);
            dialogues++;
        }
        if(count<0||(size_t)count>=CORPUS_LIMIT+1-total){fputs("corpus exceeds 16MiB\n",stderr);goto done;}
        total+=(size_t)count;rows++;
    }
    if(ferror(src)||rows<5){fputs("snapshot read error or fewer than 5 records\n",stderr);goto done;}
    fprintf(stdout,"records=%zu lessons=%zu dialogues=%zu context_retained=%zu observations_retained=%zu\n",
            rows,lessons,dialogues,context,observations);
    if(lessons+dialogues<5){
        fputs("DATA QUALITY FAILED: fewer than five usable lesson/dialogue records.\n",stderr);
        goto done;
    }
    /* Exclusive creation on MSVC: x-mode is supported. */
    FILE *out=fopen(argv[3],"wbx");
    if(!out){perror("new corpus");goto done;}
    int ok=fwrite(corpus,1,total,out)==total;
    if(fclose(out)!=0)ok=0;
    if(!ok){fputs("corpus write failure\n",stderr);remove(argv[3]);goto done;}
    printf("migration=approved_digit_jsonl records=%zu output_bytes=%zu\n",rows,total);
    result=0;
done:
    free(corpus);fclose(src);return result;
}
