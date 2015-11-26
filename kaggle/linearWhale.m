
%slope = [];
for i = 1:size(positiveInstances, 2)
    row = [];
    col = [];
   
    box = positiveInstances(i).objectBoundingBoxes;
    s = strsplit(positiveInstances(i).imageFilename, '\');
    try
        img = imread(strcat('ImagesTotal/',num2str(cell2mat(s(end)))));
    catch
        img = imread(strcat('ImagesTest/',num2str(cell2mat(s(end)))));
    end
    %img = imread(strcat('imgs/',num2str(cell2mat(s(end)))));
    %gray_img = rgb2gray(img);
    try
       img = img(box(2):box(2)+box(4), box(1):box(1)+box(3), 1:3);
    catch
        i
        if box(2)+box(4) > size(img, 1)
            img = img(box(2):size(img, 1), box(1):box(1)+box(3), 1:3);
        elseif box(1)+box(3) > size(img, 2)
            img = img(box(2):box(2)+box(4), box(1):size(img, 2), 1:3);
        end
    end
    %figure;
    %imagesc(gray_img);
    gray_img = rgb2gray(img);
    average = sum(sum(gray_img)) / (size(gray_img, 1) * size(gray_img, 2));
    
    for j = 1:size(gray_img, 1)
    for k = 1:size(gray_img, 2)
        %for k = 1:size(a, 3)
            if gray_img(j, k) <= average + 10 || gray_img(j, k) >= 200
                gray_img(j, k) = 0;
            else
                gray_img(j, k) = 255;
                num = (size(gray_img, 1) - j);
                row = [row num];
                %num = (size(gray_img, 2) - k)
                col = [col k];
            end
    end
    end
    %figure;
    %imshow(gray_img);
    %figure;
    linearCoef = polyfit(col,row,1);
    linearFit = polyval(linearCoef,col);
   % plot(col,row,'s', col,linearFit,'r-', [0 247],[0 0],'k:', [0 247],[1 1],'k:')
   %plot(col,row,'*') 
   %h = lsline;
   %set(h, 'color', 'r')
   p = polyfit(col, row, 1);
   angle = atan(p(1)) * 90;
   img_rotate= imrotate(img,angle,'bilinear','loose');
    
   imwrite(img_rotate, num2str(cell2mat(strcat('ImagesRotate/', s(end), '.jpg'))));
   slope = [slope p(1)];
   %p2 = polyfit(get(h,'xdata'),get(h,'ydata'),1);
   % mdl = LinearModel.fit(mat2dataset(double(gray_img)));
   % figure;
   % plot(mdl);
end