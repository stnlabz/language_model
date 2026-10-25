#define main regression_main
#include "../native_regression.c"
#undef main
int main(void){json_tests();printf("JSON regression failures=%d\n",failures);return failures!=0;}
