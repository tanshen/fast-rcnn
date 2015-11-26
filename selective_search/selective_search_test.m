image_db = '/home/coldmanck/kaggle';
image_filenames = textread([image_db '/data/ImageSets/test.txt'], '%s', 'delimiter', '\n');
for i = 1:length(image_filenames)
    if exist([image_db '/ImagesTest/' image_filenames{i} '.jpg'], 'file') == 2
	image_filenames{i} = [image_db '/ImagesTest/' image_filenames{i} '.jpg'];
    end
    if exist([image_db '/ImagesTest/' image_filenames{i} '.png'], 'file') == 2
        image_filenames{i} = [image_db '/ImagesTest/' image_filenames{i} '.png'];
    end
end
selective_search_rcnn(image_filenames, 'output_test.mat');
