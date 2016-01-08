%make train.txt
image_db = '/home/future/codes/fast-rcnn/data/kaggle';
s=textread([image_db '/data/ImageSets/train0.txt'],'%s');
fin=fopen([image_db '/data/ImageSets/train.txt'],'w');

for i=1:size(s)
    fprintf(fin,'%s\n',s{i});
end
for i=1:size(s)
    fprintf(fin,'%s\n',[s{i},'_r']);
end
