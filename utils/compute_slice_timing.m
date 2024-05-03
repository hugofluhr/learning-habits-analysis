% Script adapted from
% https://neurostars.org/t/heudiconv-no-extraction-of-slice-timing-data-based-on-philips-dicoms/2201/12

TRsec = 2.33384;
nSlices = 40;
TA = TRsec/nSlices; %assumes no temporal gap between volumes
bidsSliceTiming=0:TA:TRsec-TA; %ascending
%report results
fprintf('“SliceTiming”: [\n');
for i = 1 : nSlices
fprintf('%g', bidsSliceTiming(i));
if (i < nSlices)
fprintf(',\n');
else
fprintf('],\n');
end
end