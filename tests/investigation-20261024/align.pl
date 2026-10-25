use strict; use warnings;
local $/; my $s=<>;
my $old='size_t start=(size_t)(rng()%(split-(size_t)n-1));';
my $new='size_t start=(size_t)(rng()%(split-(size_t)n-1));
   /* Diagnostic: align short examples to their record boundary. */
   while(start>0 && !(start>=2 && data[start-1]==\'\\n\' && data[start-2]==\'\\n\'))start--;
   size_t end=start;
   while(end+1<split && !(data[end]==\'\\n\' && data[end+1]==\'\\n\'))end++;
   if(end-start<(size_t)n)n=(int)(end-start);
   if(n<1){sample--;continue;}';
index($s,$old)>=0 or die 'missing sampling code';
$s =~ s/\Q$old\E/$new/;
print $s;
