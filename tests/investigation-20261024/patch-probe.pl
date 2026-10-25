use strict;use warnings;local $/;
open my $f,'<','tests/native_regression.c' or die $!;my $s=<$f>;close $f;
my $start=index($s,'/* Optional controlled English');my $end=index($s,'int main(int argc');
$start>=0&&$end>$start or die 'probe markers missing';
my $new='/* Native model training and unfiltered deterministic qualification.
 * The shell runner builds C tools and keeps fixtures inside the repository. */
static int language_probe(void){
 char *args[]={"/bin/sh","./tests/language_qualification.sh",NULL};
 return run("/bin/sh",args);
}
';
open my $out,'>','tests/native_regression.c' or die $!;print $out substr($s,0,$start),$new,substr($s,$end);close $out;
