for i = 213:size(positiveInstances, 2)
    box = positiveInstances(i).objectBoundingBoxes;
    s = strsplit(positiveInstances(i).imageFilename, '\');
    try
        img = imread(strcat('ImagesTotal/',num2str(cell2mat(s(end)))));
    catch
        img = imread(strcat('ImagesTest/',num2str(cell2mat(s(end)))));
    end
    try
        img = img(box(2):box(2)+box(4), box(1):box(1)+box(3), 1:3);
    catch
        i
        if box(2)+box(4) > size(img, 1)
            img = img(box(2):size(img, 1), box(1):box(1)+box(3), 1:3);
        end
        if box(1)+box(3) > size(img, 2)
            img = img(box(2):box(2)+box(4), box(1):size(img, 2), 1:3);
        end
    end
    angle = atan(slope(i)) * 90;
    img_rotate= imrotate(img,angle,'bilinear','crop');
    
    imwrite(img_rotate, num2str(cell2mat(strcat('ImagesRotate/', s(end), '.jpg'))));
    %figure;
    %imshow(img_rotate);
end