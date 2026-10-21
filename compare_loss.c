/* Inspect validation_loss fields in eval logs and compare held-out results. */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
static int loss(const char *path,double *value) {
    char line[256],extra;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    int found=0;
    while(fgets(line,sizeof(line),f)) {
        double x;
        if(sscanf(line,"validation_loss=%lf %c",&x,&extra)==1 && isfinite(x)){
            *value=x;found=1;
        }
    }
    fclose(f);return found;
}
int main(int argc,char **argv) {
    double baseline,candidate;
    if(argc!=3||!loss(argv[1],&baseline)||!loss(argv[2],&candidate)){
        fputs("missing evaluation or invalid loss\n",stderr);return 2;
    }
    printf("reference_loss=%.6f candidate_loss=%.6f improved=%s\n",
           baseline,candidate,candidate<baseline?"yes":"no");
    return candidate<baseline?0:1;
}
