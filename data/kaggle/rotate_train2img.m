image_db = '/home/future/codes/fast-rcnn/data/kaggle';
image_filenames = textread([image_db '/data/ImageSets/train0.txt'], '%s', 'delimiter', '\n');
for i = 1:length(image_filenames)
    if exist([image_db '/data/Images/' image_filenames{i} '.jpg'], 'file') == 2
	image_filenames{i} = [image_db '/data/Images/' image_filenames{i} '.jpg'];
    end
    im=imread(image_filenames{i});
    angle=270;
    im_r=imrotate(im,angle,'bilinear','loose');
    im_r=im_r(:,end:-1:1,:);
    imwrite(im_r,[image_filenames{i}(1:end-4) '_r.jpg']);
end

