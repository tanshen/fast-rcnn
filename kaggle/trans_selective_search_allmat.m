load('test.mat');
boxes_temp = all_boxes;
fileNames = textread('data/ImageSets/test.txt', '%s');

for i = 1:size(boxes_temp,2)
    boxes_temp = uint16(all_boxes{1,i});
    boxes = boxes_temp(:, [2 1 4 3]) - 1;
    % save(['train_bbox/', fileNames{i,1} '.mat'], 'boxes');
end

all_boxes = boxes;
save('test_revised.mat', 'all_boxes');
