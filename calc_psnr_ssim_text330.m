%修改为自己的路径
GT_path='./dataset/test/total_crop';
pre_path='./dataset/results/save/res';
output_dir='./dataset/results/save';
save_dir='./dataset/results/save/res_matlab.txt';
list_path='./dataset/test/total_crop.txt';

test_scale = 4; 


compute_ifc = 0;            % IFC calculation is slow, enable when needed

fid = fopen(save_dir,'wt');

%% load image list
list_filename = fullfile(list_path);
img_list = load_list(list_filename);
num_img = length(img_list);

%% testing
PSNR = zeros(num_img, 1);
SSIM = zeros(num_img, 1);
IFC  = zeros(num_img, 1);

for i = 1:num_img
    
    img_name = img_list{i};
%     fprintf('Testing LapSRN on %s %dx: %d/%d: %s\n', dataset, test_scale, i, num_img, img_name);
    
    %% Load GT image
    GT_filename = fullfile(GT_path, sprintf('%s.png', img_name));
    img_GT = im2double(imread(GT_filename));
    img_GT = mod_crop(img_GT, test_scale);

    %% Load pre image
    pre_filename = fullfile(pre_path, sprintf('%s.png', img_name));
    img_pre = im2double(imread(pre_filename));
    img_pre = mod_crop(img_pre, test_scale);
    
    %% save result
    output_filename = fullfile(output_dir, sprintf('%s.png', img_name));
    
    %% evaluate
    [PSNR(i), SSIM(i), IFC(i)] = evaluate_SR(img_GT, img_pre, test_scale, compute_ifc);
    fprintf('Save %s\n psnr: %f\n ssim:%f\n ifc:%f\n', output_filename,PSNR(i), SSIM(i), IFC(i));
    fprintf(fid,'Save %s\n psnr: %f\n ssim:%f\n ifc:%f\n', output_filename,PSNR(i), SSIM(i), IFC(i));
    
end

PSNR(end+1) = mean(PSNR);
SSIM(end+1) = mean(SSIM);
IFC(end+1)  = mean(IFC);

fprintf('Average ----------------------- \n' );
fprintf('Average PSNR = %f\n', PSNR(end));
fprintf('Average SSIM = %f\n', SSIM(end));
fprintf('Average IFC = %f\n', IFC(end));
fprintf(fid,'Average ----------------------- \n' );
fprintf(fid,'Average PSNR = %f\n', PSNR(end));
fprintf(fid,'Average SSIM = %f\n', SSIM(end));
fprintf(fid,'Average IFC = %f\n', IFC(end));
fclose(fid);


