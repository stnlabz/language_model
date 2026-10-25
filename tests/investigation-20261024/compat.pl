use strict;use warnings;local $/;
open my $old,'<','tests/investigation-20261024/micro_llm.original.c' or die $!;my $s=<$old>;close $old;
my $start=index($s,'/* Constrain decoding');my $end=index($s,'int main(int argc');
my $legacy=substr($s,$start,$end-$start);$legacy=~s/static int generate\(/static int generate_legacy(/;
$legacy="/* Compatibility only: legacy filtered sampler used by the existing service.\n * Qualification uses explicit unfiltered greedy/sample modes below. */\n".$legacy;
open my $in,'<','micro_llm.c' or die $!;my $new=<$in>;close $in;
$new=~s@/\* Decode the actual@$legacy/* Decode the actual@;
$new=~s/if\(\(argc==6\|\|argc==7\)/if(argc==6\&\&!strcmp(argv[1],"generate"))return generate_legacy(argv[2],argv[3],atoi(argv[4]),atoi(argv[5]));\n if((argc==7)/;
open my $out,'>','micro_llm.c' or die $!;print $out $new;close $out or die $!;
