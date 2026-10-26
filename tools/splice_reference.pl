# Replaces the rows of nodes in docs/workflow-node-reference.md with freshly rendered ones.
#
#   GH_WRITE_REFERENCE=build/reference-rows.md build/bin/x64/Release/GraphicsHost.Tests.exe "[reference-rows]"
#   perl tools/splice_reference.pl build/reference-rows.md docs/workflow-node-reference.md
#
# Every node in the rendered file whose id is already in the document has its block of rows (the row with the node's name
# and the rows under it) replaced. Nodes the document does not know are listed, to be pasted into the right section by hand.
use strict;
use warnings;

my ($rowsFile, $docFile) = @ARGV;
die "usage: splice_reference.pl rows.md doc.md\n" unless $rowsFile && $docFile;

sub slurpLines {
    my ($file) = @_;
    open my $in, '<:raw', $file or die "$file: $!";
    local $/;
    my $text = <$in>;
    close $in;
    my $crlf = $text =~ /\r\n/ ? 1 : 0;
    $text =~ s/\r\n/\n/g;
    return ([split /\n/, $text, -1], $crlf);
}

sub nodeId {
    my ($line) = @_;
    return $line =~ /^\| (?!\s*\|)(.*)\(`([^`]+)`\)\s*\|/ ? $2 : undef;
}

sub isContinuation { return $_[0] =~ /^\|\s*\|/ }

my ($rows) = slurpLines($rowsFile);
my %block;
my @order;
my $current;
for my $line (@$rows) {
    next unless $line =~ /^\|/;
    next if $line =~ /^\| Node \(`id`\)/ || $line =~ /^\|---/;
    if (my $id = nodeId($line)) {
        $current = $id;
        push @order, $id;
        $block{$id} = [$line];
    } elsif (isContinuation($line) && defined $current) {
        push @{$block{$current}}, $line;
    }
}

my ($doc, $crlf) = slurpLines($docFile);
my (@out, %done);
for (my $i = 0; $i < @$doc; $i++) {
    my $line = $doc->[$i];
    my $id = nodeId($line);
    if (defined $id && $block{$id}) {
        push @out, @{$block{$id}};
        $done{$id} = 1;
        $i++ while $i + 1 < @$doc && isContinuation($doc->[$i + 1]);
    } else {
        push @out, $line;
    }
}

open my $fh, '>:raw', $docFile or die "$docFile: $!";
print $fh join($crlf ? "\r\n" : "\n", @out);
close $fh;

print "replaced: $_\n" for grep { $done{$_} } @order;
print "not in the document (add by hand): $_\n" for grep { !$done{$_} } @order;
