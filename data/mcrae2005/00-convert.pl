#!/usr/bin/perl

use strict;

my %H;
my %F;

while(<>) {
	chomp;
	next if /^Concept/;
	my ($w, $fn) = split(/\,/, $_);
	$H{$w} = {} unless defined $H{$w};
	$H{$w}->{$fn} = 1;
	$F{$fn} = {} unless defined $F{$fn};
	$F{$fn}->{$w} = 1;
}

my @words = sort (keys %H);
my %w2idx;
for(my $i = 0; $i < scalar(@words); $i++) {
	$w2idx{$words[$i]} = $i;
}

my @norms = sort(keys %F);
my %n2idx;
for(my $i = 0; $i < scalar(@norms); $i++) {
	$n2idx{$norms[$i]} = $i;
}

{
	open(my $fo, ">all_words.txt") or die $!."\n";
	print $fo join("\n", @words)."\n";
	close($fo);
}

open(my $fo, ">caracteristics.txt") or die $!."\n";
foreach my $norm (@norms) {
	my @v = (0) x scalar(@words);
	foreach my $w (keys %{ $F{$norm} } ) {
		my $idx = $w2idx{$w};
		$v[$idx] = 1;
	}
	print $fo "$norm,".join(",", @v)."\n"
}
close($fo);
