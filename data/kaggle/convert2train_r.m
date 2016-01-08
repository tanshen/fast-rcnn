clear all;
train=load('train.mat');
for j=1:2679
    boxes=train.all_boxes{j};
    clear boxes_r;
    for i=1:size(boxes)
        bx=boxes(i,:);
        boxes_r(i,1)=bx(2);
        boxes_r(i,2)=bx(1);
        boxes_r(i,3)=bx(4);
        boxes_r(i,4)=bx(3);
    end
    all_boxes{j}=boxes;
    all_boxes{j+2679}=boxes_r;    
end 
save('train_r.mat','all_boxes');
