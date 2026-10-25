use strict; use warnings;local $/;
open my $f,'<','micro_llm.c' or die $!;my $s=<$f>;close $f;
open my $d,'<','tests/investigation-20261024/decoder.txt' or die $!;my $decoder=<$d>;close $d;
my $at=index($s,'/* Constrain decoding');$at>=0 or die 'decoder marker missing';
open my $out,'>','micro_llm.c' or die $!;print $out substr($s,0,$at),$decoder;close $out or die $!;
