s=textread('./data/ImageSets/train.txt','%s');
train=load('train.mat');
for j=2680:size(s)
    boxes=train.all_boxes{j};
for i=1:size(boxes)
    bx=boxes(i,:);
    boxes(i,2)=max(0,round(bx(2)-1));
    boxes(i,1)=round(bx(1)-1);
    boxes(i,4)=max(0,round(bx(4)-1));
    boxes(i,3)=round(bx(3)-1);
end
save(strcat('./train_bbox_r/',s{j}, '_boxes.mat'),'boxes');
end
% for i=1:2
%     save('boxes',train.all_boxes{i});
% end
